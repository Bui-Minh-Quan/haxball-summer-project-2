import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl_2.model import ActorCritic
from src.rl_2.obs import ACTOR_OBS_DIM, CRITIC_STATE_DIM
from src.rl_2.pool import SelfPlayPool


def train_mappo(
    envs,
    model: ActorCritic,
    device: torch.device,
    team_size: int = 1,
    total_timesteps: int = 10_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    update_epochs: int = 2,
    minibatch_size: int = 512,
    lr_init: float = 3e-4,
    lr_final: float = 1e-5,
    ent_coef_init: float = 0.015,
    ent_coef_final: float = 0.001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 100_000,
    eval_episodes: int = 40,
    active_tiers: list[str] | None = None,
    target_tier: str = "random",                          
    filter_thresholds: dict[str, float] | None = None,    
    goal_height: float | None = None,
    save_dir: str = "models/stage1",
    pool_dir: str | None = None,
    pitch_width: float = 840.0,
    pitch_height: float = 500.0,
    max_steps: int = 900,
):
    """Clean, production-grade Vectorized MAPPO loop with Centralized Critic."""
    os.makedirs(save_dir, exist_ok=True)
    # Initialize pool (defaults to save_dir/pool if not provided)
    effective_pool_dir = pool_dir or os.path.join(save_dir, "pool")
    pool = SelfPlayPool(effective_pool_dir)
    eval_tiers = active_tiers or ["random"]

    optimizer = optim.Adam(model.parameters(), lr=lr_init, eps=1e-5)

    # 1. Dimensions and Multi-Agent Buffer Allocation
    n_agents_step = num_envs * team_size
    batch_size = num_steps * n_agents_step

    obs_buf = torch.zeros((num_steps, n_agents_step, ACTOR_OBS_DIM), device=device)
    state_buf = torch.zeros((num_steps, n_agents_step, CRITIC_STATE_DIM), device=device)
    actions_buf = torch.zeros((num_steps, n_agents_step, 2), device=device)
    logprobs_buf = torch.zeros((num_steps, n_agents_step), device=device)
    rewards_buf = torch.zeros((num_steps, n_agents_step), device=device)
    dones_buf = torch.zeros((num_steps, n_agents_step), device=device)
    values_buf = torch.zeros((num_steps, n_agents_step), device=device)

    # 2. Initial Reset
    next_payload, _ = envs.reset()
    next_obs_np = next_payload["obs"]
    next_state_np = next_payload["state"]

    # Flatten across agents if multi-agent
    if team_size == 1:
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
    else:
        next_obs = torch.as_tensor(next_obs_np.reshape(-1, ACTOR_OBS_DIM), dtype=torch.float32, device=device)
        next_state = torch.as_tensor(np.repeat(next_state_np, team_size, axis=0), dtype=torch.float32, device=device)

    next_done = torch.zeros(n_agents_step, device=device)

    global_step = 0
    next_eval_step = eval_freq
    start_time = time.time()

    print(
        f"🚀 MAPPO Initialized | Format: {team_size}v{team_size} | Envs: {num_envs} | "
        f"Step Batch: {batch_size} | Device: {device}"
    )

    # 3. Main Training Loop
    while global_step < total_timesteps:
        # Anneal Learning Rate and Entropy
        progress = global_step / float(total_timesteps)
        curr_lr = lr_init + progress * (lr_final - lr_init)
        curr_ent = ent_coef_init + progress * (ent_coef_final - ent_coef_init)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        # ── Rollout Collection ──
        for step in range(num_steps):
            global_step += n_agents_step

            obs_buf[step] = next_obs
            state_buf[step] = next_state
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs, state=next_state)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Route actions back into Gym envs
            action_np = action.cpu().numpy()
            if team_size == 1:
                env_action = action_np
            else:
                env_action = action_np.reshape(num_envs, team_size, 2)

            next_payload, reward_np, terms, truncs, _ = envs.step(env_action)
            next_obs_np = next_payload["obs"]
            next_state_np = next_payload["state"]
            next_dones_np = np.logical_or(terms, truncs)

            # Broadcast team scalar reward to all teammates
            if team_size > 1 and reward_np.size == num_envs:
                reward_np = np.repeat(reward_np, team_size)
                next_dones_np = np.repeat(next_dones_np, team_size)

            rewards_buf[step] = torch.as_tensor(reward_np.flatten(), dtype=torch.float32, device=device)

            if team_size == 1:
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
            else:
                next_obs = torch.as_tensor(next_obs_np.reshape(-1, ACTOR_OBS_DIM), dtype=torch.float32, device=device)
                next_state = torch.as_tensor(np.repeat(next_state_np, team_size, axis=0), dtype=torch.float32, device=device)

            next_done = torch.as_tensor(next_dones_np.flatten(), dtype=torch.float32, device=device)

        # ── Generalized Advantage Estimation (GAE) ──
        with torch.no_grad():
            next_value = model.get_value(next_state)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]

                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values_buf

        # ── Minibatch Optimization ──
        b_obs = obs_buf.reshape((-1, ACTOR_OBS_DIM))
        b_states = state_buf.reshape((-1, CRITIC_STATE_DIM))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Normalize advantages
        adv_std = b_advantages.std()
        if adv_std > 1e-4:
            b_advantages = (b_advantages - b_advantages.mean()) / (adv_std + 1e-8)
        else:
        # Dry batch: zero out advantages so the policy takes zero gradient steps
            b_advantages = torch.zeros_like(b_advantages)

        b_indices = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(b_indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = b_indices[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_idx],
                    state=b_states[mb_idx],
                    action=b_actions[mb_idx],
                )

                ratio = (newlogprob - b_logprobs[mb_idx]).exp()
                mb_adv = b_advantages[mb_idx]

                # Policy Loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_idx]) ** 2).mean()

                # Combined Objective
                loss = pg_loss - curr_ent * entropy.mean() + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Update latest snapshot in self-play pool
        if pool:
            pool.save_latest(model)

        # ── Evaluation & Gatekeeper Promotion ──
        if global_step >= next_eval_step:
            next_eval_step += eval_freq
            elapsed = time.time() - start_time
            sps = int(global_step / max(1.0, elapsed))

            print(f"\n📊 [EVALUATION @ Step {global_step:,} | SPS: {sps} | Tiers: {eval_tiers}]")

            passed, eval_metrics, recorded_score = pool.run_gatekeeper_gauntlet(
                learner_model=model,
                active_tiers=eval_tiers,
                target_tier=target_tier,
                filter_thresholds=filter_thresholds,
                team_size=team_size,
                goal_height=goal_height,
                pitch_width=pitch_width,
                pitch_height=pitch_height,
                device=device,
                num_episodes=eval_episodes,
                max_steps=max_steps,
            )

            for tier_name, res in eval_metrics.items():
                is_filter = tier_name in (filter_thresholds or {})
                is_target = tier_name == target_tier
                role_tag = "[TARGET]" if is_target else ("[FILTER]" if is_filter else "")
                print(
                    f"   ⚔️  vs {tier_name.capitalize():<9} {role_tag:<8} | WR: {res['win_rate']*100:5.1f}% | "
                    f"Reward: {res['mean_reward']:+.3f} | Goals: {res['scored']} Scored, {res['conceded']} Conceded ({res['net']:+d} Net)"
                )

            cand = eval_metrics.get(target_tier, {})

            if passed:
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                pool.register_champion(model, global_step)

                prev_wr = f"{recorded_score[0]*100:.1f}%" if recorded_score[0] >= 0 else "None"
                prev_rew = f"{recorded_score[1]:+.3f}" if recorded_score[0] >= 0 else "None"
                print(
                    f"   ⭐⭐ PROMOTED! New Best Score ({target_tier}) -> [WR: {cand['win_rate']*100:.1f}%, Reward: {cand['mean_reward']:+.3f}, Net: {cand['net']:+d}]\n"
                    f"      (Defeated previous record: [WR: {prev_wr}, Reward: {prev_rew}]) -> Saved: {save_path}"
                )
            else:
                best_wr = f"{recorded_score[0]*100:.1f}%" if recorded_score[0] >= 0 else "None"
                best_rew = f"{recorded_score[1]:+.3f}" if recorded_score[0] >= 0 else "None"
                best_net = f"{recorded_score[2]:+d}" if recorded_score[0] >= 0 else "None"
                print(
                    f"   ❌ Retaining current baseline. Did not pass criteria for {target_tier}: [WR: {best_wr}, Reward: {best_rew}, Net: {best_net}]"
                )
                
    # Save final model state
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n🏁 Training Complete! Final weights saved to: {final_path}")