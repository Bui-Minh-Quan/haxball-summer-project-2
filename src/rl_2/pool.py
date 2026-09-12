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
from src.rl_2.env_adapter import RandomOpponentController
from src.rl_2.model import ActorCritic
from src.rl_2.obs import extract_actor_obs


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
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-1.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
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
        ckpt = torch.load(
            target_file, map_location=self.device, weights_only=False
        )
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        actor_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("critic")
        }
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
    obs_tensor = torch.as_tensor(
        obs, dtype=torch.float32, device=self.device
    ).unsqueeze(0)

    with torch.no_grad():
      action, _, _, _ = self.model.get_action_and_value(
          obs_tensor, deterministic=True
      )

    m_idx = int(action[0, 0].item())
    kick = bool(action[0, 1].item())
    ego_x, ego_y = self._ego_dirs[m_idx]

    return Vec2(ego_x * self.sign, ego_y), kick


class SelfPlayPool:
  """Manages snapshots, champions, and gauntlet promotions for continuous play."""

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
      team_size: int = 1,
      opp_team_size: int | None = None,
      goal_height: float | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      max_steps: int = 2700,
      device: torch.device = torch.device("cpu"),
      eval_seed: int = 42,
  ) -> dict:
    """Evaluates arbitrary matchups (1v1, 2v2, 2v3, 1v3) across Random, Heuristic, and RL bots."""
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

    wins, losses, draws = 0, 0, 0
    total_scored, total_conceded = 0, 0
    ep_rewards = []

    _ego_dirs = [
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-1.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
    ]

    class LocalPlaceholder(Controller):

      def __init__(self):
        self.action = (Vec2(0, 0), False)

      def get_action(self, idx, sim):
        return self.action

    heur_coord_red = TeamHeuristicCoordinator(team="red")
    heur_coord_blue = TeamHeuristicCoordinator(team="blue")
    heur_ctrl_red = HeuristicBotController(heur_coord_red)
    heur_ctrl_blue = HeuristicBotController(heur_coord_blue)

    # Defaults to symmetric matchup if opp_team_size is omitted (Stage 1 compatible)
    l_size = team_size
    o_size = opp_team_size if opp_team_size is not None else team_size

    def apply_eval_restart(sim, ep_idx: int, is_initial: bool = False):
      """Universal asymmetric squad positioner with multi-lane defensive depth."""
      p = sim.pitch
      safe_m = 50.0

      if (ep_idx % 5 == 0) and is_initial:
        bx, by = sim.center.x, sim.center.y
        dist = min(140.0, p.width * 0.16)
        angle = 0.0
      else:
        sweep = (
            ep_idx
            if is_initial
            else (ep_idx + int(sim.score_red + sim.score_blue) * 7)
        )
        dist = min(220.0, p.width * 0.25) * (0.70 + (sweep % 4) * 0.08)
        angle = -math.pi / 4 + ((sweep * 31.0) % 90.0) * (math.pi / 180.0)
        bx = sim.center.x + (((sweep % 3) - 1) * (p.width * 0.12))
        by = sim.center.y + ((((sweep // 3) % 3) - 1) * (p.height * 0.14))

      sim.ball.pos = Vec2(bx, by)
      sim.ball.vel = Vec2(0.0, 0.0)

      vx, vy = dist * math.cos(angle), dist * math.sin(angle)
      rx_lead, ry_lead = bx - vx, by - vy
      bx_lead, by_lead = bx + vx, by + vy

      # Stagger Red Squad across defensive lanes
      for idx, pl in enumerate(sim.red_team):
        if idx == 0:
          px = max(p.left + safe_m, min(p.right - safe_m, rx_lead))
          py = max(p.top + safe_m, min(p.bottom - safe_m, ry_lead))
        else:
          back_offset = min(
              240.0, max(100.0, (rx_lead - p.left) * 0.40)
          ) * (1.0 + (idx - 1) * 0.4)
          px = max(p.left + safe_m, rx_lead - back_offset)
          lane_sign = 1.0 if (idx % 2 == 1) else -1.0
          y_offset = lane_sign * (65.0 + (idx // 2) * 55.0)
          py = min(
              max(p.top + safe_m, sim.center.y + y_offset),
              p.bottom - safe_m,
          )
        pl.pos = Vec2(px, py)
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0

      # Stagger Blue Squad across defensive lanes
      for idx, pl in enumerate(sim.blue_team):
        if idx == 0:
          px = max(p.left + safe_m, min(p.right - safe_m, bx_lead))
          py = max(p.top + safe_m, min(p.bottom - safe_m, by_lead))
        else:
          back_offset = min(
              240.0, max(100.0, (p.right - bx_lead) * 0.40)
          ) * (1.0 + (idx - 1) * 0.4)
          px = min(p.right - safe_m, bx_lead + back_offset)
          lane_sign = -1.0 if (idx % 2 == 1) else 1.0
          y_offset = lane_sign * (65.0 + (idx // 2) * 55.0)
          py = min(
              max(p.top + safe_m, sim.center.y + y_offset),
              p.bottom - safe_m,
          )
        pl.pos = Vec2(px, py)
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0

      if hasattr(sim, "mode"):
        sim.mode.state = "PLAYING"
        if hasattr(sim.mode, "celebration_timer"):
          sim.mode.celebration_timer = 0.0

    for ep in range(num_episodes):
      learner_team = "red" if ep % 2 == 0 else "blue"
      opp_team = "blue" if learner_team == "red" else "red"
      sign = 1.0 if learner_team == "red" else -1.0

      learner_placeholders = [LocalPlaceholder() for _ in range(l_size)]
      opp_placeholders = [LocalPlaceholder() for _ in range(o_size)]

      roster = []
      for i in range(l_size):
        roster.append(
            PlayerSlot(
                learner_team,
                PlayerStats(f"L{i+1}", accel=3200.0),
                learner_placeholders[i],
            )
        )
      for j in range(o_size):
        roster.append(
            PlayerSlot(
                opp_team,
                PlayerStats(f"O{j+1}", accel=3200.0),
                opp_placeholders[j],
            )
        )

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

      sim.score_red = 0
      sim.score_blue = 0

      apply_eval_restart(sim, ep_idx=ep, is_initial=True)

      physics_steps = 0
      ep_rew = 0.0
      action_repeat = 10

      while physics_steps < max_steps:
        # 1. Update Learner Actions 
        l_squad = sim.red_team if learner_team == "red" else sim.blue_team
        for idx, player in enumerate(l_squad):
          obs = extract_actor_obs(sim, player, learner_team)
          obs_t = torch.as_tensor(
              obs, dtype=torch.float32, device=device
          ).unsqueeze(0)
          with torch.no_grad():
            act, _, _, _ = learner_model.get_action_and_value(
                obs_t, deterministic=True
            )
          m_idx = int(act[0, 0].item())
          ex, ey = _ego_dirs[m_idx]
          learner_placeholders[idx].action = (
              Vec2(ex * sign, ey),
              bool(act[0, 1].item()),
          )

        # 2. Update Opponent Actions (15 Hz - Supports Random, Heuristic, and RL Models)
        o_squad = sim.blue_team if learner_team == "red" else sim.red_team
        for idx, opp_player in enumerate(o_squad):
          global_idx = sim.all_players.index(opp_player)

          if opponent_type == "heuristic":
            bot = heur_ctrl_blue if opp_team == "blue" else heur_ctrl_red
            opp_placeholders[idx].action = bot.get_action(global_idx, sim)

          elif opponent_type == "model" and opponent_model:
            obs = extract_actor_obs(sim, opp_player, opp_team)
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
              act, _, _, _ = opponent_model.get_action_and_value(
                  obs_t, deterministic=True
              )
            m_idx = int(act[0, 0].item())
            ex, ey = _ego_dirs[m_idx]
            opp_placeholders[idx].action = (
                Vec2(ex * -sign, ey),
                bool(act[0, 1].item()),
            )

          else:
            opp_placeholders[idx].action = (
                random.choice([Vec2(dx, dy) for dx, dy in _ego_dirs]),
                random.random() < 0.20,
            )

        # 3. Physics Substeps
        for _ in range(action_repeat):
          physics_steps += 1
          goal = sim.step(1.0 / 60.0)

          if hasattr(sim, "mode"):
            sim.mode.state = "PLAYING"

          if goal is not None:
            scored = goal == f"{learner_team}_goal"
            ep_rew += 1.5 if scored else -1.0
            apply_eval_restart(sim, ep_idx=ep, is_initial=False)
            break

          if physics_steps >= max_steps:
            break

      scored = sim.score_red if learner_team == "red" else sim.score_blue
      conceded = sim.score_blue if learner_team == "red" else sim.score_red
      diff = scored - conceded

      if diff > 0:
        ep_rew += 1.0 + 0.1 * diff
        wins += 1
      elif diff < 0:
        ep_rew -= 1.0 + 0.1 * abs(diff)
        losses += 1
      else:
        ep_rew -= 0.5
        draws += 1

      total_scored += scored
      total_conceded += conceded
      ep_rewards.append(ep_rew)

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
      team_size: int = 1,
      opp_team_size: int | None = None,
      goal_height: float | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      num_episodes: int = 40,
      max_steps: int = 2700,
      device: torch.device = torch.device("cpu"),
  ) -> tuple[bool, dict, tuple]:
    """Runs qualification filters across active tiers supporting custom opponent team sizes."""
    results = {}
    filters = filter_thresholds or {}

    effective_opp_size = (
        opp_team_size if opp_team_size is not None else team_size
    )

    # 1. Tier: Random Bot
    if "random" in active_tiers:
      results["random"] = self.evaluate_matchup(
          learner_model,
          opponent_type="random",
          team_size=team_size,
          opp_team_size=effective_opp_size,
          goal_height=goal_height,
          pitch_width=pitch_width,
          pitch_height=pitch_height,
          device=device,
          num_episodes=num_episodes,
          max_steps=max_steps,
      )
      if (
          "random" in filters
          and results["random"]["win_rate"] < filters["random"]
      ):
        return False, results, self.best_score

    # 2. Tier: Heuristic Bot
    if "heuristic" in active_tiers:
      results["heuristic"] = self.evaluate_matchup(
          learner_model,
          opponent_type="heuristic",
          team_size=team_size,
          opp_team_size=effective_opp_size,
          goal_height=goal_height,
          pitch_width=pitch_width,
          pitch_height=pitch_height,
          device=device,
          num_episodes=num_episodes,
          max_steps=max_steps,
      )
      if (
          "heuristic" in filters
          and results["heuristic"]["win_rate"] < filters["heuristic"]
      ):
        return False, results, self.best_score

    # 3. Tier: Champion (Self-Play)
    if "champion" in active_tiers:
      if os.path.exists(self.champion_path):
        champ = ActorCritic().to(device)
        ckpt = torch.load(
            self.champion_path, map_location=device, weights_only=False
        )
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
            num_episodes=num_episodes,
            max_steps=max_steps,
        )
      else:
        results["champion"] = {
            "win_rate": 1.0,
            "mean_reward": 1.0,
            "net": 1,
            "score_tuple": (1.0, 1.0, 1),
        }

      if (
          "champion" in filters
          and results["champion"]["win_rate"] < filters["champion"]
      ):
        return False, results, self.best_score

    # 4. Promotion Criteria
    cand = results[target_tier]
    cand_score = cand["score_tuple"]

    if target_tier == "champion":
      is_promoted = cand["win_rate"] >= 0.30 and cand["net"] >= 5
      return is_promoted, results, cand_score

    if cand_score > self.best_score:
      prev_score = self.best_score
      self.best_score = cand_score
      return True, results, prev_score

    return False, results, self.best_score