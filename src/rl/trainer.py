import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.rl.ppo_core import ActorCritic


def evaluate_policy_benchmark(
    eval_env,
    model: ActorCritic,
    device: torch.device,
    num_episodes: int = 50,
    base_seed: int = 42,
) -> dict:
    """
    Runs a deterministic benchmark across fixed sequential seeds.
    No stochastic action sampling — uses argmax greedy decisions only.
    """
    model.eval()
    goals = 0
    touches = 0
    total_rewards = []
    total_steps = []

    for ep in range(num_episodes):
        # Deterministic seed guarantees identical initial conditions on every eval run
        obs, _ = eval_env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        step_count = 0
        ep_scored = False
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
            if info.get("touched", False):
                ep_touched = True

            done = term or trunc

        if ep_scored:
            goals += 1
        if ep_touched:
            touches += 1
        total_rewards.append(ep_reward)
        total_steps.append(step_count)

    model.train()
    return {
        "goal_rate": (goals / num_episodes) * 100.0,
        "touch_rate": (touches / num_episodes) * 100.0,
        "mean_reward": float(np.mean(total_rewards)),
        "avg_steps": float(np.mean(total_steps)),
        "goals": goals,
        "total_episodes": num_episodes,
    }


def train_ppo_vectorized(
    envs,
    eval_env,
    model: ActorCritic,
    device: torch.device,
    total_timesteps: int = 1_000_000,
    num_envs: int = 16,
    num_steps: int = 256,
    eval_freq: int = 50_000,
    eval_episodes: int = 50,
    save_dir: str = "models/stage1",
    lr_initial: float = 3e-4,
    lr_final: float = 1e-5,
    ent_coef_initial: float = 0.01,
    ent_coef_final: float = 0.001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, eps=1e-5)

    obs = torch.zeros((num_steps, num_envs, 68)).to(device)
    actions = torch.zeros((num_steps, num_envs, 2)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_eval_step = eval_freq
    best_eval_goal_rate = 0.0

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    print(f"🚀 Training with Deterministic Validation ({eval_episodes} eps every {eval_freq} steps)...")

    while global_step < total_timesteps:
        # 1. Learning Rate & Entropy Annealing
        progress = global_step / total_timesteps
        current_lr = lr_initial + progress * (lr_final - lr_initial)
        current_ent_coef = ent_coef_initial + progress * (ent_coef_final - ent_coef_initial)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 2. Collect Training Rollouts (Exploratory / Stochastic)
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
        b_obs = obs.reshape((-1, 68))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 5. PPO Policy Update
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

        # 6. Deterministic Validation Gate
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
                f"Goal Rate: {metrics['goal_rate']:5.1f}% ({metrics['goals']}/{metrics['total_episodes']}) | "
                f"Touch Rate: {metrics['touch_rate']:5.1f}% | "
                f"Avg Steps: {metrics['avg_steps']:5.1f} | "
                f"Mean Reward: {metrics['mean_reward']:6.2f}"
            )

            # Strict validation gate: Only save when the deterministic benchmark improves
            if metrics["goal_rate"] >= best_eval_goal_rate:
                best_eval_goal_rate = metrics["goal_rate"]
                save_path = os.path.join(save_dir, "best_stage1.pt")
                torch.save(model.state_dict(), save_path)
                print(f"   ⭐ New verified best model saved: {save_path} ({best_eval_goal_rate:.1f}% goals)\n")

    final_path = os.path.join(save_dir, "final_stage1.pt")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training completed. Final model saved to {final_path}")