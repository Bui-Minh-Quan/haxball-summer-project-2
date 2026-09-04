import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add these two distribution imports
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.env_wrapper import RandomController
from src.rl.obs_extractor import extract_obs, extract_global_state
from src.rl.ppo_core import ActorCritic
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
    """Evaluates the team policy without logging gradients."""
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
    learner_teams, opp_teams, signs = [], [], []

    is_model_baseline = (baseline_type == "model" and baseline_model is not None)

    # --- ENVIRONMENT SETUP ---
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
        for p in range(team_size):
            ph = EvalActionPlaceholder()
            learner_placeholders[i].append(ph)
            roster.append(PlayerSlot(learner_team, PlayerStats(f"L{p}", accel=3200.0), ph))

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

    # --- BUFFER ALLOCATION ---
    state_dim = 32
    obs_batch_learner = np.zeros((total_episodes * team_size, 80), dtype=np.float32)
    state_batch_learner = np.zeros((total_episodes * team_size, state_dim), dtype=np.float32)
    
    if is_model_baseline:
        obs_batch_opp = np.zeros((total_episodes * team_size, 80), dtype=np.float32)
        state_batch_opp = np.zeros((total_episodes * team_size, state_dim), dtype=np.float32)

    ep_goals_scored = np.zeros(total_episodes, dtype=np.int32)
    ep_goals_conceded = np.zeros(total_episodes, dtype=np.int32)
    ep_rewards = np.zeros(total_episodes, dtype=np.float32)

    dt = 1.0 / 60.0

    # --- SIMULATION LOOP ---
    for step in range(max_steps):
        truncated = step == (max_steps - 1)
        idx_l, idx_o = 0, 0
        
        for i in range(total_episodes):
            l_squad = sims[i].red_team if learner_teams[i] == "red" else sims[i].blue_team
            o_squad = sims[i].blue_team if learner_teams[i] == "red" else sims[i].red_team

            # Extract Centralized State for this env
            global_state_l = extract_global_state(sims[i], learner_teams[i])
            if is_model_baseline:
                global_state_o = extract_global_state(sims[i], opp_teams[i])

            # Populate Learner Arrays
            for agent in l_squad:
                obs_batch_learner[idx_l] = extract_obs(sims[i], agent, learner_teams[i])
                state_batch_learner[idx_l] = global_state_l  # Duplicated for both agents on team
                idx_l += 1

            # Populate Opponent Arrays
            if is_model_baseline:
                for agent in o_squad:
                    obs_batch_opp[idx_o] = extract_obs(sims[i], agent, opp_teams[i])
                    state_batch_opp[idx_o] = global_state_o
                    idx_o += 1

        # Evaluate Policy (MAPPO pass)
        obs_tensor_l = torch.as_tensor(obs_batch_learner, dtype=torch.float32, device=device)
        state_tensor_l = torch.as_tensor(state_batch_learner, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            actions_l, _, _, _ = model.get_action_and_value(obs_tensor_l, state=state_tensor_l, deterministic=True)
            actions_np_l = actions_l.cpu().numpy().reshape(total_episodes, team_size, 2)

            if is_model_baseline:
                obs_tensor_o = torch.as_tensor(obs_batch_opp, dtype=torch.float32, device=device)
                state_tensor_o = torch.as_tensor(state_batch_opp, dtype=torch.float32, device=device)
                actions_o, _, _, _ = baseline_model.get_action_and_value(obs_tensor_o, state=state_tensor_o, deterministic=True)
                actions_np_o = actions_o.cpu().numpy().reshape(total_episodes, team_size, 2)

        # Apply Actions & Step Physics
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

            r, _, _ = reward_shapers[i].compute_reward(sims[i], goal_event, truncated)
            ep_rewards[i] += r

            if goal_event == f"{learner_teams[i]}_goal":
                ep_goals_scored[i] += 1
                reset_strats[i].reset(sims[i])
                reward_shapers[i].reset(sims[i])
            elif goal_event is not None:
                ep_goals_conceded[i] += 1
                reset_strats[i].reset(sims[i])
                reward_shapers[i].reset(sims[i])

    model.train()
    total_scored = int(np.sum(ep_goals_scored))
    total_conceded = int(np.sum(ep_goals_conceded))
    scored_matches = int(np.sum(ep_goals_scored > 0))
    wins = int(np.sum(ep_goals_scored > ep_goals_conceded))
    losses = int(np.sum(ep_goals_scored < ep_goals_conceded))

    return {
        "net_goals": total_scored - total_conceded,
        "total_scored": total_scored,
        "total_conceded": total_conceded,
        "scored_matches": scored_matches,
        "mean_reward": float(np.mean(ep_rewards)),
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
    **kwargs,
):
    """Centralized Training with Decentralized Execution (MAPPO) Loop with Two-Speed

    Optimization, Critic Warm-Up, and KL Regularization.

    Configurable kwargs:
        - warmup_steps (int): Steps to freeze Actor and calibrate Critic (default: 1_500_000).
        - lr_actor_initial (float): Initial learning rate for Actor fine-tuning (default: 5e-6).
        - lr_actor_final (float): Final annealed learning rate for Actor (default: 1e-6).
        - lr_critic_initial (float): Fast initial learning rate for fresh Critic (default: 3e-4).
        - lr_critic_final (float): Final annealed learning rate for Critic (default: 1e-5).
        - ppo_epochs_warmup (int): PPO epochs per batch during Critic warm-up (default: 2).
        - ppo_epochs_post (int): PPO epochs per batch during fine-tuning (default: 1).
        - minibatch_size (int): Minibatch chunk size (default: 256).
        - kl_coef (float): Weight of KL divergence penalty pulling toward reference model (default: 0.05).
        - actor_clip (float): Gradient clipping norm for Actor layers (default: 0.2).
        - critic_clip (float): Gradient clipping norm for Critic layers (default: 1.0).
        - pretrained_model_path (str | None): Path to Stage 3 champion weights for KL anchor.
        - total_timesteps (int): Total training steps (default: 15_000_000).
        - num_envs (int): Number of parallel environments (default: 16).
        - num_steps (int): Rollout steps per environment before update (default: 256).
        - eval_freq (int): Step interval between evaluations (default: 100_000).
        - eval_episodes (int): Matches per evaluation run (default: 50).
        - baseline_type (str): Opponent type for evaluation ("heuristic" or "random").
        - double_eval (bool): Whether to run secondary champion challenge (default: False).
        - save_dir (str): Checkpoint output directory (default: "models/stage4").
        - pool_dir (str | None): Live self-play pool directory (default: None).
    """
    # ── 1. Kwargs Resolution ──
    warmup_steps = kwargs.get("warmup_steps", 1_500_000)
    total_timesteps = kwargs.get("total_timesteps", 15_000_000)
    num_envs = kwargs.get("num_envs", 16)
    num_steps = kwargs.get("num_steps", 256)
    minibatch_size = kwargs.get("minibatch_size", 256)
    ppo_epochs_warmup = kwargs.get("ppo_epochs_warmup", 2)
    ppo_epochs_post = kwargs.get("ppo_epochs_post", 1)

    # Learning Rates & Regularization
    lr_actor_init = kwargs.get("lr_actor_initial", 5e-6)
    lr_actor_final = kwargs.get("lr_actor_final", 1e-6)
    lr_critic_init = kwargs.get("lr_critic_initial", 3e-4)
    lr_critic_final = kwargs.get("lr_critic_final", 1e-5)
    kl_coef = kwargs.get("kl_coef", 0.05)
    actor_clip = kwargs.get("actor_clip", 0.2)
    critic_clip = kwargs.get("critic_clip", 1.0)

    # Standard PPO Parameters
    gamma = kwargs.get("gamma", 0.99)
    gae_lambda = kwargs.get("gae_lambda", 0.95)
    clip_range = kwargs.get("clip_range", 0.2)
    ent_coef_init = kwargs.get("ent_coef_initial", 0.0005)
    ent_coef_final = kwargs.get("ent_coef_final", 0.0001)

    # Checkpoint & Eval Config
    eval_freq = kwargs.get("eval_freq", 100_000)
    eval_episodes = kwargs.get("eval_episodes", 50)
    baseline_type = kwargs.get("baseline_type", "heuristic")
    double_eval = kwargs.get("double_eval", False)
    pretrained_model_path = kwargs.get("pretrained_model_path", None)
    save_dir = kwargs.get("save_dir", "models/stage4")
    pool_dir = kwargs.get("pool_dir", None)

    # ── 2. Storage & Directory Setup ──
    os.makedirs(save_dir, exist_ok=True)
    if pool_dir:
        os.makedirs(pool_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))

    # Optional Double-Eval Opponent Model
    eval_opp_model = None
    if double_eval:
        eval_opp_model = ActorCritic(obs_dim=80, state_dim=32).to(device)
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            eval_opp_model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device, weights_only=False)
            )
        else:
            eval_opp_model.load_state_dict(model.state_dict())
        eval_opp_model.eval()

    # ── 3. Reference Model for KL Policy Anchoring ──
    ref_model = None
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        ref_model = ActorCritic(obs_dim=80, state_dim=32).to(device)
        ckpt = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        actor_state = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
        ref_model.load_state_dict(actor_state, strict=False)
        ref_model.eval()
        print(f"⚓ KL Anchor Active: Policy regularized against Stage 3 champion: {pretrained_model_path}")

    # ── 4. Two-Speed Optimizer Setup ──
    optimizer = optim.Adam([
        {"params": model.actor_encoder.parameters(), "lr": lr_actor_init, "name": "actor"},
        {"params": model.actor_move.parameters(), "lr": lr_actor_init, "name": "actor"},
        {"params": model.actor_kick.parameters(), "lr": lr_actor_init, "name": "actor"},
        {"params": model.critic.parameters(), "lr": lr_critic_init, "name": "critic"},
    ], eps=1e-5)

    # ── 5. Buffer Allocation ──
    n_agents = num_envs * team_size
    batch_size = num_steps * n_agents
    state_dim = getattr(model, "state_dim", 32)

    obs = torch.zeros((num_steps, n_agents, 80), device=device)
    states = torch.zeros((num_steps, n_agents, state_dim), device=device)
    actions = torch.zeros((num_steps, n_agents, 2), device=device)
    logprobs = torch.zeros((num_steps, n_agents), device=device)
    rewards = torch.zeros((num_steps, n_agents), device=device)
    dones = torch.zeros((num_steps, n_agents), device=device)
    values = torch.zeros((num_steps, n_agents), device=device)

    next_payload, _ = envs.reset()
    next_obs_np, next_state_np = next_payload

    next_obs = torch.as_tensor(next_obs_np.reshape(-1, 80), dtype=torch.float32, device=device)
    next_state = torch.as_tensor(np.repeat(next_state_np, team_size, axis=0), dtype=torch.float32, device=device)
    next_done = torch.zeros(n_agents, device=device)

    best_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    global_step = 0
    next_eval_step = eval_freq
    warmup_finished_alert = False

    print(
        f"🚀 MAPPO Training | Format: {team_size}v{team_size} | Envs: {num_envs} | Batch: {batch_size}\n"
        f"⚡ Two-Speed Optimizer: Actor LR = {lr_actor_init:.1e} -> {lr_actor_final:.1e} | "
        f"Critic LR = {lr_critic_init:.1e} -> {lr_critic_final:.1e}\n"
        f"🛡️  Warm-up: {warmup_steps:,} steps | KL Penalty Weight: {kl_coef} | Post-Warmup Epochs: {ppo_epochs_post}"
    )

    # ── 6. Main Training Loop ──
    while global_step < total_timesteps:
        progress = global_step / max(1, total_timesteps)
        is_warmup = global_step < warmup_steps

        if not is_warmup and not warmup_finished_alert and warmup_steps > 0:
            print(f"\n🔥 [Step {global_step:,}] Critic Warm-Up complete! Unfreezing Actor for fine-tuning updates.")
            warmup_finished_alert = True

        # Anneal Learning Rates across groups
        for param_group in optimizer.param_groups:
            if param_group.get("name") == "actor":
                param_group["lr"] = lr_actor_init + progress * (lr_actor_final - lr_actor_init)
            elif param_group.get("name") == "critic":
                param_group["lr"] = lr_critic_init + progress * (lr_critic_final - lr_critic_init)

        current_ent = ent_coef_init + progress * (ent_coef_final - ent_coef_init)

        # ── Rollout Collection ──
        for step in range(num_steps):
            global_step += n_agents
            obs[step] = next_obs
            states[step] = next_state
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs, state=next_state)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            env_action = action.cpu().numpy().reshape(num_envs, team_size, 2)
            next_payload, reward_np, terms, truncs, _ = envs.step(env_action)
            next_obs_np, next_state_np = next_payload
            next_done_np = np.logical_or(terms, truncs)

            if reward_np.size == num_envs:
                reward_np = np.repeat(reward_np, team_size)

            rewards[step] = torch.as_tensor(reward_np.reshape(-1), dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np.reshape(-1, 80), dtype=torch.float32, device=device)
            next_state = torch.as_tensor(np.repeat(next_state_np, team_size, axis=0), dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np.repeat(team_size), dtype=torch.float32, device=device)

        # ── Generalized Advantage Estimation (GAE) ──
        with torch.no_grad():
            _, _, _, next_value = model.get_action_and_value(next_obs, state=next_state)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - (next_done if t == num_steps - 1 else dones[t + 1])
                nextvalues = next_value.flatten() if t == num_steps - 1 else values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # ── PPO Minibatch Updates ──
        b_obs = obs.reshape((-1, 80))
        b_states = states.reshape((-1, state_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(batch_size)
        epochs_to_run = ppo_epochs_warmup if is_warmup else ppo_epochs_post

        for _ in range(epochs_to_run):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Critic forward pass
                newvalue = model.critic(b_states[mb_inds]).squeeze(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                if is_warmup:
                    loss = v_loss
                else:
                    # Actor forward pass
                    feat_actor = model.actor_encoder(b_obs[mb_inds])
                    logits_m = model.actor_move(feat_actor)
                    logits_k = model.actor_kick(feat_actor)

                    dist_m = Categorical(logits=logits_m)
                    dist_k = Categorical(logits=logits_k)
                    act_m = b_actions[mb_inds][:, 0]
                    act_k = b_actions[mb_inds][:, 1]

                    newlogprob = dist_m.log_prob(act_m) + dist_k.log_prob(act_k)
                    entropy = dist_m.entropy() + dist_k.entropy()

                    ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                    mb_advantages = b_advantages[mb_inds]

                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                    ).mean()

                    # KL Divergence Penalty vs Frozen Stage 3 Reference
                    kl_penalty = 0.0
                    if ref_model is not None and kl_coef > 0:
                        with torch.no_grad():
                            ref_feat = ref_model.actor_encoder(b_obs[mb_inds])
                            ref_dist_m = Categorical(logits=ref_model.actor_move(ref_feat))
                            ref_dist_k = Categorical(logits=ref_model.actor_kick(ref_feat))

                        kl_m = kl_divergence(ref_dist_m, dist_m).mean()
                        kl_k = kl_divergence(ref_dist_k, dist_k).mean()
                        kl_penalty = kl_coef * (kl_m + kl_k)

                    effective_ent = max(current_ent, ent_coef_final)
                    loss = pg_loss + kl_penalty - effective_ent * entropy.mean() + v_loss

                optimizer.zero_grad()
                loss.backward()

                # Decoupled Gradient Clipping
                if not is_warmup:
                    nn.utils.clip_grad_norm_(model.actor_encoder.parameters(), actor_clip)
                    nn.utils.clip_grad_norm_(model.actor_move.parameters(), actor_clip)
                    nn.utils.clip_grad_norm_(model.actor_kick.parameters(), actor_clip)

                nn.utils.clip_grad_norm_(model.critic.parameters(), critic_clip)
                optimizer.step()

        if pool_dir:
            torch.save(model.state_dict(), os.path.join(pool_dir, "latest.pt"))

        # ── Evaluation & Promotion ──
        if global_step >= next_eval_step:
            next_eval_step += eval_freq

            warmup_tag = " [WARM-UP]" if is_warmup else ""
            print(f"\n📊 [EVALUATION @ Step {global_step:7d}{warmup_tag} | Target: {baseline_type} {team_size}v{team_size}]")
            metrics = evaluate_team(
                model=model,
                device=device,
                baseline_type=baseline_type,
                team_size=team_size,
                num_episodes=eval_episodes,
            )

            wr = metrics["win_rate"] * 100.0
            scored_matches = metrics["scored_matches"]
            net = metrics["net_goals"]
            mean_rew = metrics["mean_reward"]
            total_eps = metrics["total_episodes"]
            wins = metrics["wins"]

            print(
                f"   ⚔️  Record : {wins}/{total_eps} Wins ({wr:5.1f}%) | Scored In: {scored_matches}/{total_eps} matches\n"
                f"   ⚽ Goals  : {metrics['total_scored']} Scored, {metrics['total_conceded']} Conceded ({net:+d} net) | Reward: {mean_rew:6.3f}"
            )

            if not double_eval:
                current_score = (metrics["win_rate"], scored_matches, net, mean_rew)

                if current_score > best_score:
                    prev_score = best_score
                    best_score = current_score

                    save_path = os.path.join(save_dir, "best_model.pt")
                    torch.save(model.state_dict(), save_path)

                    if pool_dir:
                        history_path = os.path.join(pool_dir, f"history_{global_step}.pt")
                        torch.save(model.state_dict(), history_path)
                        print(f"   📦 POOL UPDATED: Cached new generational champion -> {history_path}")

                    prev_wr = f"{prev_score[0]*100:.1f}%" if prev_score[0] != -float("inf") else "N/A"
                    prev_sc = f"{prev_score[1]}" if prev_score[1] != -float("inf") else "N/A"
                    prev_net = f"{int(prev_score[2]):+d}" if prev_score[2] != -float("inf") else "N/A"
                    prev_rew = f"{prev_score[3]:.3f}" if prev_score[3] != -float("inf") else "N/A"

                    print(
                        f"   ⭐⭐ PROMOTED! New Best Score -> Saved: {save_path}\n"
                        f"      [WR: {wr:5.1f}% (was {prev_wr}) | Scored: {scored_matches} (was {prev_sc}) | "
                        f"Net: {net:+d} (was {prev_net}) | Rew: {mean_rew:.3f} (was {prev_rew})]"
                    )
                else:
                    best_wr = f"{best_score[0]*100:.1f}%" if best_score[0] != -float("inf") else "N/A"
                    best_sc = f"{best_score[1]}" if best_score[1] != -float("inf") else "N/A"
                    best_net = f"{int(best_score[2]):+d}" if best_score[2] != -float("inf") else "N/A"
                    best_rew = f"{best_score[3]:.3f}" if best_score[3] != -float("inf") else "N/A"

                    print(
                        f"   ❌ RETAINING CURRENT BEST. Current score did not beat: "
                        f"[WR: {best_wr}, Scored: {best_sc}, Net: {best_net}, Rew: {best_rew}]"
                    )

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print(f"🏁 Training Complete! Saved final model to {save_dir}/final_model.pt")