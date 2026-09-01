import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.env_wrapper import RandomController
from src.rl.obs_extractor import extract_obs
from src.rl.reset_strategies import RandomReset
from src.rl.reward_shapers import DenseReward, BallChaserReward


class EvalActionPlaceholder(Controller):
    def __init__(self):
        self.action = (Vec2(0, 0), False)

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        return self.action


def evaluate_benchmark(
    model: nn.Module,
    device: torch.device,
    baseline_type: str = "random",
    num_episodes: int = 50,
    time_limit: float = 30.0,
    max_steps: int = 1800,
    base_seed: int = 70000,
) -> dict:
    model.eval()

    _EGO_DIRS = [
        (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
        (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
        (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
    ]

    half_episodes = num_episodes // 2
    total_episodes = half_episodes * 2

    sims = []
    reward_shapers = []
    reset_strats = []
    placeholders = []
    learner_teams = []
    signs = []

    for i in range(total_episodes):
        is_red = i < half_episodes
        learner_team = "red" if is_red else "blue"
        opp_team = "blue" if is_red else "red"
        seed = base_seed + (i if is_red else 10000 + (i - half_episodes))

        ph = EvalActionPlaceholder()
        placeholders.append(ph)
        learner_teams.append(learner_team)
        signs.append(1.0 if is_red else -1.0)

        if baseline_type == "random":
            opp_ctrl = RandomController()
        elif baseline_type == "heuristic":
            opp_ctrl = HeuristicBotController(TeamHeuristicCoordinator(opp_team))
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

        if is_red:
            roster = [
                PlayerSlot("red", PlayerStats("Learner", accel=3200.0), ph),
                PlayerSlot("blue", PlayerStats("Opponent", accel=3200.0), opp_ctrl),
            ]
        else:
            roster = [
                PlayerSlot("red", PlayerStats("Opponent", accel=3200.0), opp_ctrl),
                PlayerSlot("blue", PlayerStats("Learner", accel=3200.0), ph),
            ]

        cfg = MatchConfig(
            mode=ClassicMatchMode(time_limit=time_limit, score_limit=99),
            roster=roster,
            time_limit=time_limit,
            score_limit=99,
        )

        sim = Simulation(match_config=cfg)
        rs = RandomReset()
        rs.set_seed(seed)
        rs.reset(sim)

        rw = DenseReward(team=learner_team)
        rw.reset(sim)

        sims.append(sim)
        reset_strats.append(rs)
        reward_shapers.append(rw)

    obs_batch = np.zeros((total_episodes, 80), dtype=np.float32)
    ep_goals_scored = np.zeros(total_episodes, dtype=np.int32)
    ep_goals_conceded = np.zeros(total_episodes, dtype=np.int32)
    ep_rewards = np.zeros(total_episodes, dtype=np.float32)
    first_goal_times = [None] * total_episodes

    dt = 1.0 / 60.0

    for step in range(max_steps):
        for i in range(total_episodes):
            agent = sims[i].red_team[0] if learner_teams[i] == "red" else sims[i].blue_team[0]
            obs_batch[i] = extract_obs(sims[i], agent, learner_teams[i])

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)

        actions_np = actions.cpu().numpy()
        truncated = step == (max_steps - 1)

        for i in range(total_episodes):
            m_idx = int(actions_np[i, 0])
            k_val = bool(actions_np[i, 1])

            # Apply sign mirroring to transform ego-action to world space
            ego_x, ego_y = _EGO_DIRS[m_idx]
            world_move = Vec2(ego_x * signs[i], ego_y)
            placeholders[i].action = (world_move, k_val)

            goal_event = sims[i].step(dt)
            r, _, _ = reward_shapers[i].compute_reward(sims[i], goal_event, truncated)
            ep_rewards[i] += r

            if goal_event == f"{learner_teams[i]}_goal":
                ep_goals_scored[i] += 1
                if first_goal_times[i] is None:
                    first_goal_times[i] = (step + 1) * dt
                reset_strats[i].reset(sims[i])
                reward_shapers[i].reset(sims[i])
            elif goal_event is not None:
                ep_goals_conceded[i] += 1
                reset_strats[i].reset(sims[i])
                reward_shapers[i].reset(sims[i])

    model.train()

    episodes_with_goals = int(np.sum(ep_goals_scored > 0))
    valid_speeds = [t for t in first_goal_times if t is not None]
    avg_speed = float(np.mean(valid_speeds)) if valid_speeds else 0.0
    total_scored = int(np.sum(ep_goals_scored))
    total_conceded = int(np.sum(ep_goals_conceded))
    net_goals = total_scored - total_conceded
    avg_reward = float(np.mean(ep_rewards))

    score_key = (net_goals, episodes_with_goals, total_scored, round(avg_reward, 2))

    return {
        "net_goals": net_goals,
        "episodes_with_goals": episodes_with_goals,
        "total_goals_scored": total_scored,
        "total_goals_conceded": total_conceded,
        "avg_speed": avg_speed,
        "avg_reward": avg_reward,
        "score_key": score_key,
        "total_episodes": total_episodes,
    }


def train_ppo(
    envs,
    model: nn.Module,
    device: torch.device,
    baseline_type: str = "random",
    total_timesteps: int = 5_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    max_steps: int = 1800,
    time_limit: float = 30.0,
    eval_freq: int = 100_000,
    eval_episodes: int = 50,
    save_dir: str = "models/stage1",
    pool_dir: str | None = None,
    lr_initial: float = 3e-4,
    lr_final: float = 1e-5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef_initial: float = 0.015,
    ent_coef_final: float = 0.0005,
):
    os.makedirs(save_dir, exist_ok=True)
    if pool_dir:
        os.makedirs(pool_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))
        torch.save(model.state_dict(), os.path.join(pool_dir, "history_0.pt"))

    optimizer = optim.Adam(model.parameters(), lr=lr_initial, eps=1e-5)
    obs_dim = envs.single_observation_space.shape[0]

    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, 2), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    next_obs_np, _ = envs.reset()
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    global_step = 0
    next_eval_step = eval_freq
    next_history_save = 500_000
    batch_size = num_steps * num_envs
    best_score_key = (-1, -999, -float("inf"))

    print(f"🚀 Training: Target [{baseline_type.upper()}] | Batch: {batch_size} | Eval: {eval_episodes} eps")

    while global_step < total_timesteps:
        progress = global_step / total_timesteps
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_initial + progress * (lr_final - lr_initial)
        current_ent = ent_coef_initial + progress * (ent_coef_final - ent_coef_initial)

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

            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device).flatten()
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, _, _, next_value = model.get_action_and_value(next_obs)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - (next_done if t == num_steps - 1 else dones[t + 1])
                nextvalues = next_value.flatten() if t == num_steps - 1 else values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(batch_size)
        for _ in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, 128):
                end = start + 128
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                mb_advantages = b_advantages[mb_inds]

                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                v_loss = 0.5 * 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                effective_entropy_coef = max(current_ent, 0.003)
                loss = pg_loss - effective_entropy_coef * entropy.mean() + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if pool_dir:
            torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))
            if global_step >= next_history_save:
                torch.save(model.state_dict(), os.path.join(pool_dir, f"history_{global_step}.pt"))
                next_history_save += 500_000

        if global_step >= next_eval_step:
            next_eval_step += eval_freq
            metrics = evaluate_benchmark(
                model=model,
                device=device,
                baseline_type=baseline_type,
                num_episodes=eval_episodes,
                time_limit=time_limit,
                max_steps=max_steps,
            )

            pct = (metrics["episodes_with_goals"] / metrics["total_episodes"]) * 100.0
            print(f"\n📊 [EVALUATION @ Step {global_step:7d} | Target: {baseline_type.upper()}]")
            print(f"   Scoring Episodes: {metrics['episodes_with_goals']}/{metrics['total_episodes']} ({pct:.1f}%)")
            print(f"   Goals [Scored: {metrics['total_goals_scored']} | Conceded: {metrics['total_goals_conceded']} | Net: {metrics['net_goals']:+d}]")
            print(f"   Avg Speed to 1st Goal: {metrics['avg_speed']:.2f}s | Mean Reward: {metrics['avg_reward']:.2f}")

            if metrics["score_key"] > best_score_key:
                best_score_key = metrics["score_key"]
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"   ⭐⭐ PROMOTED! New Best Score: {best_score_key} -> Saved: {save_path}")
            else:
                print(f"   ❌ Retaining best model. (Best Score: {best_score_key})")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))