import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.rl.benchmarker import RLController


def evaluate_policy_benchmark(
    eval_env,
    model: ActorCritic,
    device: torch.device,
    num_episodes: int = 50,
    base_seed: int = 1000,
) -> dict:
    model.eval()
    goals_scored = 0
    goals_conceded = 0
    touches = 0
    total_rewards = []
    total_steps = []

    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        step_count = 0
        ep_scored = False
        ep_conceded = False
        ep_touched = False

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)

            obs, reward, term, trunc, info = eval_env.step(action.squeeze(0).cpu().numpy())
            ep_reward += reward
            step_count += 1

            if info.get("is_goal", False):
                ep_scored = True
            if info.get("conceded", False):
                ep_conceded = True
            if info.get("touched", False):
                ep_touched = True

            done = term or trunc

        if ep_scored:
            goals_scored += 1
        if ep_conceded:
            goals_conceded += 1
        if ep_touched:
            touches += 1

        total_rewards.append(ep_reward)
        total_steps.append(step_count)

    model.train()
    return {
        "scored_rate": (goals_scored / num_episodes) * 100.0,
        "conceded_rate": (goals_conceded / num_episodes) * 100.0,
        "touch_rate": (touches / num_episodes) * 100.0,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "net_diff": goals_scored - goals_conceded,
        "mean_reward": float(np.mean(total_rewards)),
        "avg_steps": float(np.mean(total_steps)),
        "total_episodes": num_episodes,
    }

def evaluate_against_baselines(
    model,
    stage1_model,
    device,
    num_matches: int = 30,
    time_limit: float = 30.0,
) -> dict:
    """Evaluates the learner policy against Stage 1 Baseline and Heuristic Bot."""
    model.eval()
    learner_ctrl = RLController(model, team="red", device=device)
    stage1_ctrl = RLController(stage1_model, team="blue", device=device)

    blue_coord = TeamHeuristicCoordinator(team="blue")
    heur_ctrl = HeuristicBotController(blue_coord)

    # 1. Match series vs Stage 1 Base Checkpoint
    cfg_s1 = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=1),
        roster=[
            PlayerSlot(
                "red", PlayerStats("Learner", accel=3200.0), learner_ctrl
            ),
            PlayerSlot(
                "blue", PlayerStats("Stage1_Base", accel=3200.0), stage1_ctrl
            ),
        ],
        time_limit=time_limit,
        score_limit=1,
    )

    s1_wins, s1_losses, s1_draws = 0, 0, 0
    for _ in range(num_matches):
        sim = Simulation(match_config=cfg_s1)
        while not sim.mode.is_game_over(sim):
            sim.step(1.0 / 60.0)
        if sim.score_red > sim.score_blue:
            s1_wins += 1
        elif sim.score_blue > sim.score_red:
            s1_losses += 1
        else:
            s1_draws += 1

    # 2. Match series vs Heuristic Bot
    cfg_heur = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=1),
        roster=[
            PlayerSlot(
                "red", PlayerStats("Learner", accel=3200.0), learner_ctrl
            ),
            PlayerSlot(
                "blue", PlayerStats("Heur_Bot", accel=3000.0), heur_ctrl
            ),
        ],
        time_limit=time_limit,
        score_limit=1,
    )

    h_wins, h_losses, h_draws = 0, 0, 0
    for _ in range(num_matches):
        sim = Simulation(match_config=cfg_heur)
        while not sim.mode.is_game_over(sim):
            sim.step(1.0 / 60.0)
        if sim.score_red > sim.score_blue:
            h_wins += 1
        elif sim.score_blue > sim.score_red:
            h_losses += 1
        else:
            h_draws += 1

    model.train()
    return {
        "vs_stage1_win_rate": (s1_wins / num_matches) * 100.0,
        "vs_stage1_loss_rate": (s1_losses / num_matches) * 100.0,
        "vs_stage1_draw_rate": (s1_draws / num_matches) * 100.0,
        "vs_heur_win_rate": (h_wins / num_matches) * 100.0,
        "vs_heur_loss_rate": (h_losses / num_matches) * 100.0,
        "vs_heur_draw_rate": (h_draws / num_matches) * 100.0,
    }

def evaluate_multi_agent_benchmark(
    eval_env, model, device, num_episodes=50, base_seed=1000
):
    model.eval()
    red_wins = 0
    blue_wins = 0
    draws = 0
    total_steps = []

    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        done = False
        step_count = 0
        goal_event = None

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(
                    obs_tensor, deterministic=True
                )

            obs, _, term, trunc, info = eval_env.step(action.cpu().numpy())
            step_count += 1
            done = term or trunc
            if info.get("goal_event"):
                goal_event = info.get("goal_event")

        total_steps.append(step_count)
        if goal_event == "red_goal":
            red_wins += 1
        elif goal_event == "blue_goal":
            blue_wins += 1
        else:
            draws += 1

    model.train()
    return {
        "red_win_rate": (red_wins / num_episodes) * 100.0,
        "blue_win_rate": (blue_wins / num_episodes) * 100.0,
        "draw_rate": (draws / num_episodes) * 100.0,
        "avg_steps": float(np.mean(total_steps)),
        "total_episodes": num_episodes,
    }


def train_ppo_vectorized(
    envs,
    eval_env,
    model: ActorCritic,
    device: torch.device,
    total_timesteps: int = 2_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    eval_freq: int = 100_000,
    eval_episodes: int = 50,
    save_dir: str = "models/stage1",
    model_name: str = "best_model",
    lr_initial: float = 2.5e-4,
    lr_final: float = 1e-5,
    ent_coef_initial: float = 0.01,
    ent_coef_final: float = 0.001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, eps=1e-5)

    # Dynamic observation space detection
    if hasattr(envs, "single_observation_space"):
        obs_dim = envs.single_observation_space.shape[0]
    else:
        obs_dim = envs.observation_space.shape[0]

    obs = torch.zeros((num_steps, num_envs, obs_dim)).to(device)
    actions = torch.zeros((num_steps, num_envs, 2)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_eval_step = eval_freq

    # Lexicographical benchmark baseline initialized outside the loop
    best_eval_key = (
        -float("inf"),  # 1. Net diff
        -float("inf"),  # 2. Scored rate
        -float("inf"),  # 3. Mean reward
        -float("inf"),  # 4. -Avg steps
        -float("inf"),  # 5. Touch rate
        -float("inf"),  # 6. -Conceded rate
    )

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    print(f"🚀 Training ({obs_dim} dims) | Benchmark every {eval_freq} steps...")

    while global_step < total_timesteps:
        # 1. Learning Rate & Entropy Annealing
        progress = global_step / total_timesteps
        current_lr = lr_initial + progress * (lr_final - lr_initial)
        current_ent_coef = ent_coef_initial + progress * (ent_coef_final - ent_coef_initial)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 2. Rollout Collection
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terms, truncs, _ = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terms, truncs)

            rewards[step] = torch.tensor(reward_np).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

        # 3. GAE Advantage Estimation
        with torch.no_grad():
            _, _, _, next_value = model.get_action_and_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 4. Flatten Batches
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 5. Mini-Batch SGD Update
        b_inds = np.arange(num_steps * num_envs)
        for _ in range(6):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 128):
                end = start + 128
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - current_ent_coef * entropy.mean() + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # 6. Benchmark Evaluation Gate & Checkpointing
        if global_step >= next_eval_step:
            next_eval_step += eval_freq
            metrics = evaluate_policy_benchmark(
                eval_env=eval_env,
                model=model,
                device=device,
                num_episodes=eval_episodes,
                base_seed=1000,
            )

            print(
                f"\n📊 [EVALUATION @ Step {global_step:7d}] "
                f"Scored: {metrics['scored_rate']:5.1f}% ({metrics['goals_scored']}/{metrics['total_episodes']}) | "
                f"Conceded: {metrics['conceded_rate']:5.1f}% ({metrics['goals_conceded']}/{metrics['total_episodes']}) | "
                f"Net: {metrics['net_diff']:+3d} | "
                f"Touch: {metrics['touch_rate']:5.1f}% | "
                f"Avg Steps: {metrics['avg_steps']:5.1f} | "
                f"Mean Reward: {metrics['mean_reward']:6.2f}"
            )

            current_eval_key = (
                metrics["net_diff"],
                metrics["scored_rate"],
                metrics["mean_reward"],
                -metrics["avg_steps"],
                metrics["touch_rate"],
                -metrics["conceded_rate"],
            )

            # Strictly evaluate and save inside the evaluation trigger
            if current_eval_key > best_eval_key:
                best_eval_key = current_eval_key
                save_path = os.path.join(save_dir, f"{model_name}.pt")
                torch.save(model.state_dict(), save_path)
                print(
                    f"   ⭐ New verified best model saved: {save_path}\n"
                    f"      [Net: {metrics['net_diff']:+d} | "
                    f"Scored: {metrics['scored_rate']:.1f}% | "
                    f"Reward: {metrics['mean_reward']:.2f} | "
                    f"Speed: {metrics['avg_steps']:.1f} steps]\n"
                )

    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training completed. Final model saved to {final_path}")


def evaluate_multi_agent_benchmark(eval_env, model, device, num_episodes=50, base_seed=1000):
    """Evaluates N vs M self-play matches and tracks team win rates."""
    model.eval()
    red_wins = 0
    blue_wins = 0
    draws = 0
    total_steps = []

    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        done = False
        step_count = 0
        goal_event = None

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)
            
            obs, rewards, term, trunc, info = eval_env.step(action.cpu().numpy())
            step_count += 1
            done = term or trunc
            if info.get("goal_event"):
                goal_event = info.get("goal_event")

        total_steps.append(step_count)
        
        if goal_event == "red_goal":
            red_wins += 1
        elif goal_event == "blue_goal":
            blue_wins += 1
        else:
            draws += 1

    model.train()
    return {
        "red_win_rate": (red_wins / num_episodes) * 100.0,
        "blue_win_rate": (blue_wins / num_episodes) * 100.0,
        "draw_rate": (draws / num_episodes) * 100.0,
        "avg_steps": float(np.mean(total_steps)),
        "total_episodes": num_episodes,
    }


def train_ppo_multi_agent(
    envs,
    model,
    stage1_model,
    device: torch.device,
    total_timesteps: int = 10_000_000,
    num_envs: int = 16,
    num_agents_per_env: int = 2,
    num_steps: int = 256,
    eval_freq: int = 100_000,
    eval_episodes: int = 30,
    save_dir: str = "models/stage3",
    lr_initial: float = 8e-5,
    lr_final: float = 5e-6,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef_initial: float = 0.005,
    ent_coef_final: float = 0.0002,
):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, eps=1e-5)

    obs_dim = envs.single_observation_space.shape[1]

    obs = torch.zeros((num_steps, num_envs, num_agents_per_env, obs_dim)).to(
        device
    )
    actions = torch.zeros((num_steps, num_envs, num_agents_per_env, 2)).to(
        device
    )
    logprobs = torch.zeros((num_steps, num_envs, num_agents_per_env)).to(device)
    rewards = torch.zeros((num_steps, num_envs, num_agents_per_env)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs, num_agents_per_env)).to(device)

    next_obs_np, _ = envs.reset()
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(num_envs).to(device)

    global_step = 0
    next_eval_step = eval_freq
    best_eval_score = -float("inf")

    print(
        f"🚀 Multi-Agent Self-Play | Batch Size: {num_envs * num_agents_per_env * num_steps}"
    )

    while global_step < total_timesteps:
        progress = global_step / total_timesteps
        current_lr = lr_initial + progress * (lr_final - lr_initial)
        current_ent_coef = ent_coef_initial + progress * (
            ent_coef_final - ent_coef_initial
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 1. Rollout Collection
        for step in range(num_steps):
            global_step += num_envs * num_agents_per_env
            obs[step] = next_obs
            dones[step] = next_done

            obs_flat = next_obs.view(-1, obs_dim)
            with torch.no_grad():
                action_flat, logprob_flat, _, val_flat = (
                    model.get_action_and_value(obs_flat)
                )

            actions[step] = action_flat.view(num_envs, num_agents_per_env, 2)
            logprobs[step] = logprob_flat.view(num_envs, num_agents_per_env)
            values[step] = val_flat.view(num_envs, num_agents_per_env)

            actions_np = actions[step].cpu().numpy()
            next_obs_np, scalar_reward_np, terms, truncs, infos = envs.step(
                actions_np
            )

            # Assign +100/-100 on goal, -50/-50 on draw timeout
            step_rewards = np.zeros(
                (num_envs, num_agents_per_env), dtype=np.float32
            )
            for e_idx in range(num_envs):
                if terms[e_idx]:  # Goal occurred
                    step_rewards[e_idx, 0] = scalar_reward_np[e_idx]
                    step_rewards[e_idx, 1] = -scalar_reward_np[e_idx]
                elif truncs[e_idx]:  # Timeout draw penalty
                    step_rewards[e_idx, 0] = -50.0
                    step_rewards[e_idx, 1] = -50.0

            next_done_np = np.logical_or(terms, truncs)
            rewards[step] = torch.tensor(step_rewards).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

        # 2. GAE Advantage Calculation
        with torch.no_grad():
            _, _, _, next_val_flat = model.get_action_and_value(
                next_obs.view(-1, obs_dim)
            )
            next_value = next_val_flat.view(num_envs, num_agents_per_env)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = torch.zeros((num_envs, num_agents_per_env)).to(device)

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = (1.0 - next_done).unsqueeze(1)
                    nextvalues = next_value
                else:
                    nextnonterminal = (1.0 - dones[t + 1]).unsqueeze(1)
                    nextvalues = values[t + 1]

                delta = (
                    rewards[t]
                    + gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # 3. Optimization
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        b_inds = np.arange(num_steps * num_envs * num_agents_per_env)
        for _ in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 256):
                end = start + 256
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - current_ent_coef * entropy.mean() + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # 4. Baseline Benchmark Checkpoint Gate
        if global_step >= next_eval_step:
            next_eval_step += eval_freq
            metrics = evaluate_against_baselines(
                model=model,
                stage1_model=stage1_model,
                device=device,
                num_matches=eval_episodes,
                time_limit=30.0,
            )

            print(
                f"\n📊 [EVALUATION @ Step {global_step:7d}] "
                f"vs Stage1: Win {metrics['vs_stage1_win_rate']:4.1f}% | Loss {metrics['vs_stage1_loss_rate']:4.1f}% | Draw {metrics['vs_stage1_draw_rate']:4.1f}% || "
                f"vs Heuristic: Win {metrics['vs_heur_win_rate']:4.1f}% | Loss {metrics['vs_heur_loss_rate']:4.1f}% | Draw {metrics['vs_heur_draw_rate']:4.1f}%"
            )

            combined_win_score = (
                metrics["vs_stage1_win_rate"] + metrics["vs_heur_win_rate"]
            )
            if combined_win_score >= best_eval_score:
                best_eval_score = combined_win_score
                save_path = os.path.join(save_dir, "best_selfplay_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"   ⭐ New best self-play model saved: {save_path}")

    torch.save(
        model.state_dict(), os.path.join(save_dir, "final_selfplay_model.pt")
    )


def train_ppo_league(
    envs,
    eval_env,
    model,
    device: torch.device,
    total_timesteps: int = 15_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    eval_freq: int = 100_000,
    eval_episodes: int = 50,
    save_dir: str = "models/stage3_league",
    pool_dir: str = "models/stage3_league/pool",
    lr_initial: float = 8e-5,
    lr_final: float = 5e-6,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef_initial: float = 0.005,
    ent_coef_final: float = 0.0002,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)
    
    # Initialize the pool with the starting model
    torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))
    torch.save(model.state_dict(), os.path.join(pool_dir, "history_0.pt"))

    optimizer = optim.Adam(model.parameters(), lr=lr_initial, eps=1e-5)
    obs_dim = envs.single_observation_space.shape[0]

    obs = torch.zeros((num_steps, num_envs, obs_dim)).to(device)
    actions = torch.zeros((num_steps, num_envs, 2)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_eval_step = eval_freq
    next_history_save = 500_000
    
    # Track the best performing model
    best_eval_key = -float('inf')

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    print(f"🚀 Fictitious Self-Play League Training Started")

    while global_step < total_timesteps:
        progress = global_step / total_timesteps
        current_lr = lr_initial + progress * (lr_final - lr_initial)
        current_ent_coef = ent_coef_initial + progress * (ent_coef_final - ent_coef_initial)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terms, truncs, _ = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terms, truncs)

            rewards[step] = torch.tensor(reward_np).to(device).flatten()
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

        with torch.no_grad():
            _, _, _, next_value = model.get_action_and_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(num_steps * num_envs)
        for _ in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 128):
                end = start + 128
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - current_ent_coef * entropy.mean() + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # -----------------------------------------------------------------
        # LEAGUE SYNC MECHANICS
        # -----------------------------------------------------------------
        # 1. Update the "Latest" model in the pool so the environments use it
        torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))

        # 2. Save a historical snapshot every 500k steps
        if global_step >= next_history_save:
            torch.save(model.state_dict(), os.path.join(pool_dir, f"history_{global_step}.pt"))
            next_history_save += 500_000

        # 3. Evaluate using evaluate_against_baselines (from previous response)
        if global_step >= next_eval_step:
            next_eval_step += eval_freq
            from src.rl.trainer import evaluate_against_baselines
            # Evaluate against the fixed historical Stage 1 and Heuristic
            metrics = evaluate_against_baselines(
                model=model,
                stage1_model=model, # Temporarily use self if no distinct stage1 model passed
                device=device,
                num_matches=eval_episodes,
                time_limit=30.0,
            )

            print(
                f"\n📊 [EVALUATION @ Step {global_step:7d}] "
                f"vs Heuristic: Win {metrics['vs_heur_win_rate']:4.1f}% | Loss {metrics['vs_heur_loss_rate']:4.1f}% | Draw {metrics['vs_heur_draw_rate']:4.1f}%"
            )

            current_score = metrics["vs_heur_win_rate"] - metrics["vs_heur_loss_rate"]
            if current_score >= best_eval_key:
                best_eval_key = current_score
                save_path = os.path.join(save_dir, "best_league_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"   ⭐ New best league model saved: {save_path}")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_league_model.pt"))