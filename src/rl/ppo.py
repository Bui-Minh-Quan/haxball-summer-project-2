import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.model import ActorCritic
from src.rl.obs import ACTOR_OBS_DIM, CRITIC_STATE_DIM
from src.rl.pool import SelfPlayPool


def train_mappo(
    envs,
    model: ActorCritic,
    device: torch.device,
    team_size: int = 3,
    opp_team_size: int | None = None,
    total_timesteps: int = 50_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    update_epochs: int = 3,
    minibatch_size: int = 1536,
    lr_init: float = 4e-5,
    lr_final: float = 2e-6,
    ent_coef_init: float = 0.007,
    ent_coef_final: float = 0.0008,
    gamma: float = 0.995,
    gae_lambda: float = 0.97,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 250_000,
    eval_episodes: int | dict[str, int] = 50,
    tier_ratios: dict[str, float] | None = None,
    active_tiers: list[str] | None = None,
    target_tier: str = "champion",
    filter_thresholds: dict[str, float] | None = None,
    goal_height: float | None = None,
    save_dir: str = "models/stage3/phase2",
    pool_dir: str | None = None,
    pitch_width: float = 1200.0,
    pitch_height: float = 800.0,
    max_steps: int = 3600,
    action_repeat: int = 10,
):
  """Accelerated MAPPO with CUDA Mixed Precision and Vectorized Evaluation."""
  os.makedirs(save_dir, exist_ok=True)
  effective_pool_dir = pool_dir or os.path.join(save_dir, "pool")
  pool = SelfPlayPool(effective_pool_dir)
  eval_tiers = active_tiers or ["heuristic", "champion"]

  optimizer = optim.Adam(model.parameters(), lr=lr_init, eps=1e-5)
  scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

  n_agents_step = num_envs * team_size
  batch_size = num_steps * n_agents_step

  obs_buf = torch.zeros((num_steps, n_agents_step, ACTOR_OBS_DIM), device=device)
  state_buf = torch.zeros((num_steps, n_agents_step, CRITIC_STATE_DIM), device=device)
  actions_buf = torch.zeros((num_steps, n_agents_step, 2), device=device)
  logprobs_buf = torch.zeros((num_steps, n_agents_step), device=device)
  rewards_buf = torch.zeros((num_steps, n_agents_step), device=device)
  dones_buf = torch.zeros((num_steps, n_agents_step), device=device)
  values_buf = torch.zeros((num_steps, n_agents_step), device=device)

  next_payload, _ = envs.reset()
  next_obs_np = next_payload["obs"]
  next_state_np = next_payload["state"]

  if team_size == 1:
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
  else:
    next_obs = torch.as_tensor(
        next_obs_np.reshape(-1, ACTOR_OBS_DIM), dtype=torch.float32, device=device
    )
    next_state = torch.as_tensor(
        np.repeat(next_state_np, team_size, axis=0), dtype=torch.float32, device=device
    )

  next_done = torch.zeros(n_agents_step, device=device)

  global_step = 0
  next_eval_step = eval_freq
  interval_start_time = time.time()
  interval_steps = 0

  print(
      f"🚀 Accelerated MAPPO Initialized | Format: {team_size}v{opp_team_size or team_size} | "
      f"Envs: {num_envs} | Batch: {batch_size} | Device: {device}"
  )

  while global_step < total_timesteps:
    progress = global_step / float(total_timesteps)
    curr_lr = lr_init + progress * (lr_final - lr_init)
    curr_ent = ent_coef_init + progress * (ent_coef_final - ent_coef_init)
    for param_group in optimizer.param_groups:
      param_group["lr"] = curr_lr

    # ── Rollout Collection (Zero-overhead Inference Mode) ──
    for step in range(num_steps):
      global_step += n_agents_step
      interval_steps += n_agents_step

      obs_buf[step] = next_obs
      state_buf[step] = next_state
      dones_buf[step] = next_done

      with torch.inference_mode():
        action, logprob, _, value = model.get_action_and_value(next_obs, state=next_state)
        values_buf[step] = value.flatten()

      actions_buf[step] = action
      logprobs_buf[step] = logprob

      action_np = action.cpu().numpy()
      env_action = action_np if team_size == 1 else action_np.reshape(num_envs, team_size, 2)

      next_payload, reward_np, terms, truncs, _ = envs.step(env_action)
      next_obs_np = next_payload["obs"]
      next_state_np = next_payload["state"]
      next_dones_np = np.logical_or(terms, truncs)

      if team_size > 1 and reward_np.size == num_envs:
        reward_np = np.repeat(reward_np, team_size)
        next_dones_np = np.repeat(next_dones_np, team_size)

      rewards_buf[step] = torch.as_tensor(
          reward_np.flatten(), dtype=torch.float32, device=device
      )

      if team_size == 1:
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
      else:
        next_obs = torch.as_tensor(
            next_obs_np.reshape(-1, ACTOR_OBS_DIM), dtype=torch.float32, device=device
        )
        next_state = torch.as_tensor(
            np.repeat(next_state_np, team_size, axis=0), dtype=torch.float32, device=device
        )

      next_done = torch.as_tensor(next_dones_np.flatten(), dtype=torch.float32, device=device)

    # ── GAE ──
    with torch.inference_mode():
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

    # ── AMP Minibatch Optimization ──
    b_obs = obs_buf.reshape((-1, ACTOR_OBS_DIM))
    b_states = state_buf.reshape((-1, CRITIC_STATE_DIM))
    b_logprobs = logprobs_buf.reshape(-1)
    b_actions = actions_buf.reshape((-1, 2))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    adv_std = b_advantages.std()
    b_advantages = (
        (b_advantages - b_advantages.mean()) / (adv_std + 1e-8)
        if adv_std > 1e-4
        else torch.zeros_like(b_advantages)
    )

    b_indices = np.arange(batch_size)
    for _ in range(update_epochs):
      np.random.shuffle(b_indices)
      for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_idx = b_indices[start:end]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
          _, newlogprob, entropy, newvalue = model.get_action_and_value(
              b_obs[mb_idx],
              state=b_states[mb_idx],
              action=b_actions[mb_idx],
          )

          ratio = (newlogprob - b_logprobs[mb_idx]).exp()
          mb_adv = b_advantages[mb_idx]

          pg_loss1 = -mb_adv * ratio
          pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
          pg_loss = torch.max(pg_loss1, pg_loss2).mean()

          v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_idx]) ** 2).mean()
          loss = pg_loss - curr_ent * entropy.mean() + vf_coef * v_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    if pool:
      pool.save_latest(model)

    # ── Fast Evaluation ──
    if global_step >= next_eval_step:
      next_eval_step += eval_freq
      interval_sps = int(interval_steps / max(1e-3, (time.time() - interval_start_time)))

      print(f"\n📊 [EVALUATION @ Step {global_step:,} | Rollout SPS: {interval_sps} | Tiers: {eval_tiers}]")

      eval_t0 = time.time()
      passed, eval_metrics, recorded_score = pool.run_gatekeeper_gauntlet(
          learner_model=model,
          active_tiers=eval_tiers,
          target_tier=target_tier,
          filter_thresholds=filter_thresholds,
          team_size=team_size,
          opp_team_size=opp_team_size,
          goal_height=goal_height,
          pitch_width=pitch_width,
          pitch_height=pitch_height,
          num_episodes=eval_episodes,
          tier_ratios=tier_ratios,
          action_repeat=action_repeat,
          device=device,
          max_steps=max_steps,
      )
      eval_duration = time.time() - eval_t0

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
            f"      (Defeated previous record: [WR: {prev_wr}, Reward: {prev_rew}]) -> Saved: {save_path} (Eval took {eval_duration:.1f}s)"
        )
      else:
        best_wr = f"{recorded_score[0]*100:.1f}%" if recorded_score[0] >= 0 else "None"
        best_rew = f"{recorded_score[1]:+.3f}" if recorded_score[0] >= 0 else "None"
        best_net = f"{recorded_score[2]:+d}" if recorded_score[0] >= 0 else "None"
        print(
            f"   ❌ Retaining current baseline. Did not pass criteria for {target_tier}: "
            f"[WR: {best_wr}, Reward: {best_rew}, Net: {best_net}] (Eval took {eval_duration:.1f}s)"
        )

      interval_start_time = time.time()
      interval_steps = 0

  final_path = os.path.join(save_dir, "final_model.pt")
  torch.save(model.state_dict(), final_path)
  print(f"\n🏁 Training Complete! Final weights saved to: {final_path}")