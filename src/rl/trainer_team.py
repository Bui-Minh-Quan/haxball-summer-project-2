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
from src.rl.reward_shapers import DenseReward_4

class EvalActionPlaceholder(Controller):
    def __init__(self):
        self.action = (Vec2(0, 0), False)
    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        return self.action

def evaluate_team(
    model: nn.Module,
    device: torch.device,
    baseline_type: str = "heuristic",
    baseline_model: nn.Module | None = None,
    num_episodes: int = 50,
    team_size: int = 2,
    time_limit: float = 30.0,
    max_steps: int = 1800,
    base_seed: int = 90000,
) -> dict:
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()

    _EGO_DIRS = [
        (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
        (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
        (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
    ]

    half_episodes = num_episodes // 2
    total_episodes = half_episodes * 2

    sims, reset_strats, reward_shapers = [], [], []
    learner_placeholders = [[] for _ in range(total_episodes)]
    opp_placeholders = [[] for _ in range(total_episodes)]
    learner_teams = []
    opp_teams = []
    signs = []

    is_model_baseline = (baseline_type == "model" and baseline_model is not None)

    for i in range(total_episodes):
        is_red = i < half_episodes
        learner_team = "red" if is_red else "blue"
        opp_team = "blue" if is_red else "red"
        match_pair_idx = i if is_red else (i - half_episodes)
        seed = base_seed + match_pair_idx

        learner_teams.append(learner_team)
        opp_teams.append(opp_team)
        signs.append(1.0 if is_red else -1.0)

        roster = []
        # Populate Learners
        for p in range(team_size):
            ph = EvalActionPlaceholder()
            learner_placeholders[i].append(ph)
            roster.append(PlayerSlot(learner_team, PlayerStats(f"L{p}", accel=3200.0), ph))
        
        # Populate Opponents
        if is_model_baseline:
            for p in range(team_size):
                o_ph = EvalActionPlaceholder()
                opp_placeholders[i].append(o_ph)
                roster.append(PlayerSlot(opp_team, PlayerStats(f"O{p}", accel=3200.0), o_ph))
        else:
            opp_ctrl = HeuristicBotController(TeamHeuristicCoordinator(opp_team)) if baseline_type == "heuristic" else RandomController()
            for p in range(team_size):
                roster.append(PlayerSlot(opp_team, PlayerStats(f"O{p}", accel=3200.0), opp_ctrl))

        cfg = MatchConfig(mode=ClassicMatchMode(time_limit=time_limit, score_limit=99), roster=roster)
        sim = Simulation(match_config=cfg)
        rs = RandomReset()
        rs.set_seed(seed)
        rs.reset(sim)
        rw = DenseReward_4(team=learner_team)
        rw.reset(sim)

        sims.append(sim)
        reset_strats.append(rs)
        reward_shapers.append(rw)

    obs_batch_learner = np.zeros((total_episodes * team_size, 80), dtype=np.float32)
    obs_batch_opp = np.zeros((total_episodes * team_size, 80), dtype=np.float32) if is_model_baseline else None

    ep_goals_scored = np.zeros(total_episodes, dtype=np.int32)
    ep_goals_conceded = np.zeros(total_episodes, dtype=np.int32)

    dt = 1.0 / 60.0

    for step in range(max_steps):
        idx_l = 0
        idx_o = 0
        for i in range(total_episodes):
            l_squad = sims[i].red_team if learner_teams[i] == "red" else sims[i].blue_team
            o_squad = sims[i].blue_team if learner_teams[i] == "red" else sims[i].red_team
            
            for agent in l_squad:
                obs_batch_learner[idx_l] = extract_obs(sims[i], agent, learner_teams[i])
                idx_l += 1
                
            if is_model_baseline:
                for agent in o_squad:
                    obs_batch_opp[idx_o] = extract_obs(sims[i], agent, opp_teams[i])
                    idx_o += 1

        obs_tensor_l = torch.as_tensor(obs_batch_learner, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions_l, _, _, _ = model.get_action_and_value(obs_tensor_l, deterministic=True)
            actions_np_l = actions_l.cpu().numpy().reshape(total_episodes, team_size, 2)

            if is_model_baseline:
                obs_tensor_o = torch.as_tensor(obs_batch_opp, dtype=torch.float32, device=device)
                actions_o, _, _, _ = baseline_model.get_action_and_value(obs_tensor_o, deterministic=True)
                actions_np_o = actions_o.cpu().numpy().reshape(total_episodes, team_size, 2)

        for i in range(total_episodes):
            for p in range(team_size):
                m_idx_l = int(actions_np_l[i, p, 0])
                ego_x_l, ego_y_l = _EGO_DIRS[m_idx_l]
                learner_placeholders[i][p].action = (Vec2(ego_x_l * signs[i], ego_y_l), bool(actions_np_l[i, p, 1]))

                if is_model_baseline:
                    m_idx_o = int(actions_np_o[i, p, 0])
                    ego_x_o, ego_y_o = _EGO_DIRS[m_idx_o]
                    opp_placeholders[i][p].action = (Vec2(ego_x_o * -signs[i], ego_y_o), bool(actions_np_o[i, p, 1]))

            goal_event = sims[i].step(dt)
            if goal_event == f"{learner_teams[i]}_goal":
                ep_goals_scored[i] += 1
                reset_strats[i].reset(sims[i])
            elif goal_event is not None:
                ep_goals_conceded[i] += 1
                reset_strats[i].reset(sims[i])

    model.train()
    total_scored = int(np.sum(ep_goals_scored))
    total_conceded = int(np.sum(ep_goals_conceded))
    wins = int(np.sum(ep_goals_scored > ep_goals_conceded))
    losses = int(np.sum(ep_goals_scored < ep_goals_conceded))
    
    return {
        "net_goals": total_scored - total_conceded,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_episodes,
        "loss_rate": losses / total_episodes,
        "total_episodes": total_episodes,
    }



def train_team_ppo(
    envs,
    model: nn.Module,
    device: torch.device,
    team_size: int = 2,
    double_eval: bool = True,
    stage3_model_path: str | None = None,
    total_timesteps: int = 15_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    eval_freq: int = 100_000,
    eval_episodes: int = 50,
    save_dir: str = "models/stage4",
    pool_dir: str | None = None,
    lr_initial: float = 3e-5,
    lr_final: float = 5e-6,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef_initial: float = 0.006,
    ent_coef_final: float = 0.002,
):
    os.makedirs(save_dir, exist_ok=True)
    if pool_dir:
        os.makedirs(pool_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))

    # Track champion in memory for double evaluation
    eval_opp_model = None
    if double_eval:
        from src.rl.ppo_core import ActorCritic
        eval_opp_model = ActorCritic(obs_dim=80).to(device)
        if stage3_model_path and os.path.exists(stage3_model_path):
            eval_opp_model.load_state_dict(torch.load(stage3_model_path, map_location=device, weights_only=False))
        else:
            eval_opp_model.load_state_dict(model.state_dict())
        eval_opp_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr_initial, eps=1e-5)

    # IPPO Flattening: Buffer scales by team_size
    n_agents = num_envs * team_size
    batch_size = num_steps * n_agents

    obs = torch.zeros((num_steps, n_agents, 80), device=device)
    actions = torch.zeros((num_steps, n_agents, 2), device=device)
    logprobs = torch.zeros((num_steps, n_agents), device=device)
    rewards = torch.zeros((num_steps, n_agents), device=device)
    dones = torch.zeros((num_steps, n_agents), device=device)
    values = torch.zeros((num_steps, n_agents), device=device)

    next_obs_np, _ = envs.reset()
    next_obs = torch.as_tensor(next_obs_np.reshape(-1, 80), dtype=torch.float32, device=device)
    next_done = torch.zeros(n_agents, device=device)

    global_step = 0
    next_eval_step = eval_freq

    print(f"🚀 Multi-Agent Training (IPPO) | Format: {team_size}v{team_size} | Envs: {num_envs} | Batch: {batch_size}")

    while global_step < total_timesteps:
        # Anneal Learning Rate and Entropy Coefficient
        progress = global_step / total_timesteps
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_initial + progress * (lr_final - lr_initial)
        current_ent = ent_coef_initial + progress * (ent_coef_final - ent_coef_initial)

        # 1. Rollout Collection
        for step in range(num_steps):
            global_step += n_agents
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Reshape actions back to (num_envs, team_size, 2) for multi-agent stepping
            env_action = action.cpu().numpy().reshape(num_envs, team_size, 2)
            next_obs_np, reward_np, terms, truncs, _ = envs.step(env_action)
            next_done_np = np.logical_or(terms, truncs)

            # Flatten transitions back into IPPO buffer
            rewards[step] = torch.as_tensor(reward_np.reshape(-1), dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np.reshape(-1, 80), dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np.repeat(team_size), dtype=torch.float32, device=device)

        # 2. Generalized Advantage Estimation (GAE) across all agents
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

        # 3. PPO Minibatch Updates
        b_obs = obs.reshape((-1, 80))
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

                # Policy Loss
                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                # Value Loss
                v_loss = 0.5 * 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                effective_ent = max(current_ent, ent_coef_final)
                loss = pg_loss - effective_ent * entropy.mean() + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # Update latest model for live pool sampling
        if pool_dir:
            torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))

        # 4. Multi-Agent Evaluation & Champion Trial
        if global_step >= next_eval_step:
            next_eval_step += eval_freq

            print(f"\n📊 [EVALUATION @ Step {global_step:7d} | Target: HEURISTIC {team_size}v{team_size}]")
            metrics_heu = evaluate_team(
                model=model,
                device=device,
                baseline_type="heuristic",
                team_size=team_size,
                num_episodes=eval_episodes,
            )

            wr_heu = metrics_heu["win_rate"] * 100.0
            print(f"   Win Rate: {wr_heu:.1f}% | Net Goals: {metrics_heu['net_goals']:+d} | Wins: {metrics_heu['wins']}/{metrics_heu['total_episodes']}")

            # Pass 1: Heuristic Sanity Check Guardrail
            guardrail_failures = []
            if metrics_heu["win_rate"] < 0.80:
                guardrail_failures.append(f"Win Rate: {wr_heu:.1f}% < 80.0%")
            if metrics_heu["net_goals"] <= 0:
                guardrail_failures.append(f"Net Goals: {metrics_heu['net_goals']:+d} ≤ 0")

            if guardrail_failures:
                reason_str = " | ".join(guardrail_failures)
                print(f"   ❌ FAILED SANITY CHECK [{reason_str}]. Skipping champion trial.")
                continue

            # Pass 2: Champion Trial
            if double_eval and eval_opp_model is not None:
                print(f"   ⚔️ [DOUBLE EVAL] Passed sanity check! Challenging Champion {team_size}v{team_size}...")
                metrics_champ = evaluate_team(
                    model=model,
                    device=device,
                    baseline_type="model",
                    baseline_model=eval_opp_model,
                    team_size=team_size,
                    num_episodes=eval_episodes,
                )

                wr = metrics_champ["win_rate"]
                lr = metrics_champ["loss_rate"]
                net = metrics_champ["net_goals"]
                win_loss_spread = (wr - lr) * 100.0

                print(f"   ⚔️ Results: Win: {wr*100:.1f}% | Loss: {lr*100:.1f}% | Spread: {win_loss_spread:+.1f}% | Net Goals: {net:+d}")

                # Verify decisive dethroning margin
                champ_failures = []
                if net < 10:
                    champ_failures.append(f"Net Goals: {net:+d} < +10")
                if (wr - lr) < 0.10:
                    champ_failures.append(f"Win-Loss Spread: {win_loss_spread:+.1f}% < +10.0%")

                dethroned = (len(champ_failures) == 0)

                if dethroned:
                    save_path = os.path.join(save_dir, "best_model.pt")
                    torch.save(model.state_dict(), save_path)
                    eval_opp_model.load_state_dict(model.state_dict())
                    print(f"   ⭐⭐ PROMOTED! Decisively defeated Champion -> Saved: {save_path}")

                    # Event-driven pool expansion
                    if pool_dir:
                        history_path = os.path.join(pool_dir, f"history_{global_step}.pt")
                        torch.save(model.state_dict(), history_path)
                        print(f"   📦 POOL UPDATED: Cached new generational champion -> {history_path}")
                else:
                    reason_str = " | ".join(champ_failures)
                    print(f"   ❌ RETAINING CHAMPION [{reason_str}].")
            else:
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"   ⭐⭐ PROMOTED! New Best Score -> Saved: {save_path}")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print(f"🏁 Training Complete! Saved final model to {save_dir}/final_model.pt")