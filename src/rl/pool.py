import glob
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.env_adapter import RandomOpponentController
from src.rl.model import ActorCritic
from src.rl.obs import extract_actor_obs


class PoolOpponentController(Controller):
  """Dynamic opponent controller sampling across Random, Heuristic, and Self-Play models."""

  def __init__(
      self,
      pool_dir: str | None = None,
      team: str = "blue",
      device: str = "cpu",
      p_random: float = 0.05,
      p_heuristic: float = 0.45,
  ):
    torch.set_num_threads(1)
    self.pool_dir = pool_dir
    self.team = team
    self.sign = 1.0 if team == "red" else -1.0
    self.device = torch.device(device)

    self.p_random = p_random
    self.p_heuristic = p_heuristic

    self.random_ctrl = RandomOpponentController()
    self.heuristic_ctrl = HeuristicBotController(
        TeamHeuristicCoordinator(team=team)
    )
    self.model = ActorCritic().to(self.device)
    self.model.eval()

    self.current_mode = "random"
    self._ego_dirs = [
        (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
        (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
        (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
    ]

  def reset_opponent(self):
    roll = random.random()

    if (
        roll < self.p_random
        or not self.pool_dir
        or not os.path.exists(self.pool_dir)
    ):
      self.current_mode = "random"
      return

    if roll < (self.p_random + self.p_heuristic):
      self.current_mode = "heuristic"
      return

    history_files = glob.glob(os.path.join(self.pool_dir, "history_*.pt"))
    latest_file = os.path.join(self.pool_dir, "latest.pt")
    target_file = None

    if history_files and random.random() < 0.50:
      target_file = random.choice(history_files)
    elif os.path.exists(latest_file):
      target_file = latest_file

    if target_file:
      try:
        ckpt = torch.load(target_file, map_location=self.device, weights_only=False)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        actor_dict = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
        self.model.load_state_dict(actor_dict, strict=False)
        self.current_mode = "model"
        return
      except Exception:
        pass

    self.current_mode = "heuristic"

  def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
    if self.current_mode == "random":
      return self.random_ctrl.get_action(player_idx, sim)
    elif self.current_mode == "heuristic":
      return self.heuristic_ctrl.get_action(player_idx, sim)

    player = sim.all_players[player_idx]
    obs = extract_actor_obs(sim, player, self.team)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    with torch.inference_mode():
      action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)

    m_idx = int(action[0, 0].item())
    kick = bool(action[0, 1].item())
    ego_x, ego_y = self._ego_dirs[m_idx]
    return Vec2(ego_x * self.sign, ego_y), kick


class SelfPlayPool:
  """Manages snapshots, champions, and high-throughput batched evaluations."""

  def __init__(self, pool_dir: str):
    self.pool_dir = pool_dir
    os.makedirs(self.pool_dir, exist_ok=True)
    self.champion_path = os.path.join(self.pool_dir, "champion.pt")
    self.best_score = (-1.0, -float("inf"), -float("inf"))

  def save_latest(self, model: nn.Module):
    torch.save(model.state_dict(), os.path.join(self.pool_dir, "latest.pt"))

  def register_champion(self, model: nn.Module, step: int):
    torch.save(model.state_dict(), self.champion_path)
    history_path = os.path.join(self.pool_dir, f"history_{step}.pt")
    torch.save(model.state_dict(), history_path)
    print(f"🏆 NEW CHAMPION REGISTERED @ step {step:,} -> {history_path}")

  def evaluate_matchup(
      self,
      learner_model: nn.Module,
      opponent_type: str,
      opponent_model: nn.Module | None = None,
      num_episodes: int = 40,
      team_size: int = 3,
      opp_team_size: int | None = None,
      goal_height: float | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      max_steps: int = 3600,
      action_repeat: int = 10,
      device: torch.device = torch.device("cpu"),
      eval_seed: int = 42,
  ) -> dict:
    """High-throughput lockstep evaluation running all matches concurrently with batched GPU inference."""
    learner_model.eval()
    if opponent_model:
      opponent_model.eval()

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
      cuda_state = torch.cuda.get_rng_state()

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    _ego_dirs = [
        (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
        (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
        (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
    ]

    class LocalPlaceholder(Controller):
      def __init__(self):
        self.action = (Vec2(0, 0), False)
      def get_action(self, idx, sim):
        return self.action

    heur_ctrl_red = HeuristicBotController(TeamHeuristicCoordinator(team="red"))
    heur_ctrl_blue = HeuristicBotController(TeamHeuristicCoordinator(team="blue"))

    l_size = team_size
    o_size = opp_team_size if opp_team_size is not None else team_size

    def apply_eval_restart(sim, ep_idx: int, is_initial: bool = False):
      p = sim.pitch
      safe_m = 50.0

      if (ep_idx % 5 == 0) and is_initial:
        bx, by = sim.center.x, sim.center.y
        dist = min(140.0, p.width * 0.16)
        angle = 0.0
      else:
        sweep = ep_idx if is_initial else (ep_idx + int(sim.score_red + sim.score_blue) * 7)
        dist = min(220.0, p.width * 0.25) * (0.70 + (sweep % 4) * 0.08)
        angle = -math.pi / 4 + ((sweep * 31.0) % 90.0) * (math.pi / 180.0)
        bx = sim.center.x + (((sweep % 3) - 1) * (p.width * 0.12))
        by = sim.center.y + ((((sweep // 3) % 3) - 1) * (p.height * 0.14))

      sim.ball.pos = Vec2(bx, by)
      sim.ball.vel = Vec2(0.0, 0.0)

      vx, vy = dist * math.cos(angle), dist * math.sin(angle)
      rx_lead, ry_lead = bx - vx, by - vy
      bx_lead, by_lead = bx + vx, by + vy

      for idx, pl in enumerate(sim.red_team):
        if idx == 0:
          px = max(p.left + safe_m, min(p.right - safe_m, rx_lead))
          py = max(p.top + safe_m, min(p.bottom - safe_m, ry_lead))
        else:
          back_offset = min(240.0, max(100.0, (rx_lead - p.left) * 0.40)) * (1.0 + (idx - 1) * 0.4)
          px = max(p.left + safe_m, rx_lead - back_offset)
          lane_sign = 1.0 if (idx % 2 == 1) else -1.0
          y_offset = lane_sign * (65.0 + (idx // 2) * 55.0)
          py = min(max(p.top + safe_m, sim.center.y + y_offset), p.bottom - safe_m)
        pl.pos = Vec2(px, py)
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0

      for idx, pl in enumerate(sim.blue_team):
        if idx == 0:
          px = max(p.left + safe_m, min(p.right - safe_m, bx_lead))
          py = max(p.top + safe_m, min(p.bottom - safe_m, by_lead))
        else:
          back_offset = min(240.0, max(100.0, (p.right - bx_lead) * 0.40)) * (1.0 + (idx - 1) * 0.4)
          px = min(p.right - safe_m, bx_lead + back_offset)
          lane_sign = -1.0 if (idx % 2 == 1) else 1.0
          y_offset = lane_sign * (65.0 + (idx // 2) * 55.0)
          py = min(max(p.top + safe_m, sim.center.y + y_offset), p.bottom - safe_m)
        pl.pos = Vec2(px, py)
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0

      if hasattr(sim, "mode"):
        sim.mode.state = "PLAYING"

    # 1. Instantiate all simulation matches simultaneously
    sims = []
    learner_phs = []
    opp_phs = []
    learner_teams = []
    opp_teams = []
    signs = []
    ep_rewards = np.zeros(num_episodes, dtype=np.float32)

    for ep in range(num_episodes):
      l_team = "red" if ep % 2 == 0 else "blue"
      o_team = "blue" if l_team == "red" else "red"
      learner_teams.append(l_team)
      opp_teams.append(o_team)
      signs.append(1.0 if l_team == "red" else -1.0)

      l_phs = [LocalPlaceholder() for _ in range(l_size)]
      o_phs = [LocalPlaceholder() for _ in range(o_size)]
      learner_phs.append(l_phs)
      opp_phs.append(o_phs)

      roster = []
      for i in range(l_size):
        roster.append(PlayerSlot(l_team, PlayerStats(f"L{i+1}", accel=3200.0), l_phs[i]))
      for j in range(o_size):
        roster.append(PlayerSlot(o_team, PlayerStats(f"O{j+1}", accel=3200.0), o_phs[j]))

      cfg = MatchConfig(
          mode=ClassicMatchMode(time_limit=max_steps / 60.0, score_limit=99),
          roster=roster,
          goal_height=goal_height,
          pitch_width=pitch_width,
          pitch_height=pitch_height,
      )
      sim = Simulation(match_config=cfg, goal_height=goal_height)
      if hasattr(sim, "mode"):
        sim.mode.state = "PLAYING"
        if hasattr(sim.mode, "reset_positions"):
          sim.mode.reset_positions = lambda *args, **kwargs: None

      apply_eval_restart(sim, ep_idx=ep, is_initial=True)
      sims.append(sim)

    total_decisions = max_steps // action_repeat

    # 2. Batched lockstep progression (all episodes step concurrently)
    for dec in range(total_decisions):
      # Extract observations for all learner agents across all active matches
      obs_l = []
      for ep in range(num_episodes):
        sim = sims[ep]
        squad = sim.red_team if learner_teams[ep] == "red" else sim.blue_team
        for pl in squad:
          obs_l.append(extract_actor_obs(sim, pl, learner_teams[ep]))

      obs_l_t = torch.as_tensor(np.array(obs_l, dtype=np.float32), device=device)
      with torch.inference_mode():
        acts_l, _, _, _ = learner_model.get_action_and_value(obs_l_t, deterministic=True)
      acts_l_np = acts_l.cpu().numpy()

      # Assign learner actions
      cursor_l = 0
      for ep in range(num_episodes):
        sign = signs[ep]
        for pl_idx in range(l_size):
          m_idx = int(acts_l_np[cursor_l, 0])
          k_val = bool(acts_l_np[cursor_l, 1])
          ex, ey = _ego_dirs[m_idx]
          learner_phs[ep][pl_idx].action = (Vec2(ex * sign, ey), k_val)
          cursor_l += 1

      # Handle opponent actions
      if opponent_type == "model" and opponent_model is not None:
        obs_o = []
        for ep in range(num_episodes):
          sim = sims[ep]
          squad = sim.blue_team if learner_teams[ep] == "red" else sim.red_team
          for pl in squad:
            obs_o.append(extract_actor_obs(sim, pl, opp_teams[ep]))

        obs_o_t = torch.as_tensor(np.array(obs_o, dtype=np.float32), device=device)
        with torch.inference_mode():
          acts_o, _, _, _ = opponent_model.get_action_and_value(obs_o_t, deterministic=True)
        acts_o_np = acts_o.cpu().numpy()

        cursor_o = 0
        for ep in range(num_episodes):
          sign = signs[ep]
          for pl_idx in range(o_size):
            m_idx = int(acts_o_np[cursor_o, 0])
            k_val = bool(acts_o_np[cursor_o, 1])
            ex, ey = _ego_dirs[m_idx]
            opp_phs[ep][pl_idx].action = (Vec2(ex * -sign, ey), k_val)
            cursor_o += 1

      elif opponent_type == "heuristic":
        for ep in range(num_episodes):
          sim = sims[ep]
          squad = sim.blue_team if learner_teams[ep] == "red" else sim.red_team
          bot = heur_ctrl_blue if opp_teams[ep] == "blue" else heur_ctrl_red
          for pl_idx, opp_pl in enumerate(squad):
            g_idx = sim.all_players.index(opp_pl)
            opp_phs[ep][pl_idx].action = bot.get_action(g_idx, sim)

      else:
        for ep in range(num_episodes):
          for pl_idx in range(o_size):
            dx, dy = random.choice(_ego_dirs)
            opp_phs[ep][pl_idx].action = (Vec2(dx, dy), random.random() < 0.20)

      # Step all matches forward
      for _ in range(action_repeat):
        for ep in range(num_episodes):
          sim = sims[ep]
          goal = sim.step(1.0 / 60.0)
          if hasattr(sim, "mode"):
            sim.mode.state = "PLAYING"
          if goal is not None:
            scored = goal == f"{learner_teams[ep]}_goal"
            ep_rewards[ep] += 1.0 if scored else -1.0
            apply_eval_restart(sim, ep_idx=ep, is_initial=False)

    # 3. Aggregate results across episodes
    wins, losses, draws = 0, 0, 0
    total_scored, total_conceded = 0, 0

    for ep in range(num_episodes):
      sim = sims[ep]
      l_team = learner_teams[ep]
      scored = sim.score_red if l_team == "red" else sim.score_blue
      conceded = sim.score_blue if l_team == "red" else sim.score_red
      diff = scored - conceded

      if diff > 0:
        ep_rewards[ep] += 1.0 + 0.1 * diff
        wins += 1
      elif diff < 0:
        ep_rewards[ep] -= 1.0 + 0.1 * abs(diff)
        losses += 1
      else:
        ep_rewards[ep] -= 0.5
        draws += 1

      total_scored += scored
      total_conceded += conceded

    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available():
      torch.cuda.set_rng_state(cuda_state)

    learner_model.train()
    mean_reward = float(np.mean(ep_rewards))
    win_rate = wins / max(1, num_episodes)
    net_goals = total_scored - total_conceded

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "mean_reward": round(mean_reward, 3),
        "scored": total_scored,
        "conceded": total_conceded,
        "net": net_goals,
        "score_tuple": (win_rate, round(mean_reward, 3), net_goals),
    }

  def run_gatekeeper_gauntlet(
      self,
      learner_model: nn.Module,
      active_tiers: list[str],
      target_tier: str = "champion",
      filter_thresholds: dict[str, float] | None = None,
      team_size: int = 3,
      opp_team_size: int | None = None,
      goal_height: float | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      num_episodes: int | dict[str, int] = 40,
      tier_ratios: dict[str, float] | None = None,
      action_repeat: int = 10,
      max_steps: int = 3600,
      device: torch.device = torch.device("cpu"),
  ) -> tuple[bool, dict, tuple]:
    """Runs qualification filters across active tiers with budgeted episodes per tier."""
    results = {}
    filters = filter_thresholds or {}
    effective_opp_size = opp_team_size if opp_team_size is not None else team_size

    # Budget episodes across tiers
    tier_ep_counts: dict[str, int] = {}
    if isinstance(num_episodes, dict):
      tier_ep_counts = num_episodes
    elif tier_ratios is not None:
      for t in active_tiers:
        r = tier_ratios.get(t, 1.0 / max(1, len(active_tiers)))
        tier_ep_counts[t] = max(1, int(round(num_episodes * r)))
    else:
      for t in active_tiers:
        tier_ep_counts[t] = num_episodes

    # 1. Tier: Random Bot
    if "random" in active_tiers:
      eps = tier_ep_counts.get("random", 10)
      results["random"] = self.evaluate_matchup(
          learner_model,
          opponent_type="random",
          team_size=team_size,
          opp_team_size=effective_opp_size,
          goal_height=goal_height,
          pitch_width=pitch_width,
          pitch_height=pitch_height,
          device=device,
          num_episodes=eps,
          max_steps=max_steps,
          action_repeat=action_repeat,
      )
      if "random" in filters and results["random"]["win_rate"] < filters["random"]:
        return False, results, self.best_score

    # 2. Tier: Heuristic Bot
    if "heuristic" in active_tiers:
      eps = tier_ep_counts.get("heuristic", 20)
      results["heuristic"] = self.evaluate_matchup(
          learner_model,
          opponent_type="heuristic",
          team_size=team_size,
          opp_team_size=effective_opp_size,
          goal_height=goal_height,
          pitch_width=pitch_width,
          pitch_height=pitch_height,
          device=device,
          num_episodes=eps,
          max_steps=max_steps,
          action_repeat=action_repeat,
      )
      if "heuristic" in filters and results["heuristic"]["win_rate"] < filters["heuristic"]:
        return False, results, self.best_score

    # 3. Tier: Champion (Self-Play)
    if "champion" in active_tiers:
      eps = tier_ep_counts.get("champion", 30)
      if os.path.exists(self.champion_path):
        champ = ActorCritic().to(device)
        ckpt = torch.load(self.champion_path, map_location=device, weights_only=False)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        champ.load_state_dict(state_dict, strict=False)
        champ.eval()

        results["champion"] = self.evaluate_matchup(
            learner_model,
            opponent_type="model",
            opponent_model=champ,
            team_size=team_size,
            opp_team_size=effective_opp_size,
            goal_height=goal_height,
            pitch_width=pitch_width,
            pitch_height=pitch_height,
            device=device,
            num_episodes=eps,
            max_steps=max_steps,
            action_repeat=action_repeat,
        )
      else:
        results["champion"] = {
            "win_rate": 1.0,
            "mean_reward": 1.0,
            "net": 1,
            "score_tuple": (1.0, 1.0, 1),
        }

      if "champion" in filters and results["champion"]["win_rate"] < filters["champion"]:
        return False, results, self.best_score

    cand = results[target_tier]
    cand_score = cand["score_tuple"]

    if target_tier == "champion":
      is_promoted = cand["win_rate"] >= 0.25 and cand["net"] >= 5
      return is_promoted, results, cand_score

    if cand_score > self.best_score:
      prev_score = self.best_score
      self.best_score = cand_score
      return True, results, prev_score

    return False, results, self.best_score