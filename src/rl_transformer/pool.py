import glob
import math
import multiprocessing as mp
import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_transformer.entity_obs import extract_entity_obs
from src.rl_transformer.env_adapter import RandomOpponentController
from src.rl_transformer.transformer_model import TransformerActorCritic


# ── Direction Action Lookup ──
_EGO_DIRS = [
    (0.0, 0.0),   # 0: None
    (0.0, -1.0),  # 1: Up
    (0.0, 1.0),   # 2: Down
    (-1.0, 0.0),  # 3: Backward
    (1.0, 0.0),   # 4: Forward
    (-1.0, -1.0), # 5: Backward-Up
    (1.0, -1.0),  # 6: Forward-Up
    (-1.0, 1.0),  # 7: Backward-Down
    (1.0, 1.0),   # 8: Forward-Down
]


class LocalPlaceholder(Controller):
  def __init__(self):
    self.action = (Vec2(0, 0), False)

  def get_action(self, idx, sim):
    return self.action


def _apply_eval_restart(sim: Simulation, ep_idx: int, is_initial: bool = False):
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


# ── Top-Level Multiprocessing Worker ──
def _eval_single_episode_worker(task: dict) -> dict:
  torch.set_num_threads(1)
  ep = task["ep"]
  ep_seed = task["eval_seed"] + ep * 1000
  random.seed(ep_seed)
  np.random.seed(ep_seed)
  torch.manual_seed(ep_seed)

  l_size = task["team_size"]
  o_size = task["opp_team_size"]
  l_team = "red" if ep % 2 == 0 else "blue"
  o_team = "blue" if l_team == "red" else "red"
  sign = 1.0 if l_team == "red" else -1.0

  # Setup CPU models
  learner_model = TransformerActorCritic()
  learner_model.load_state_dict(task["learner_state_dict"], strict=False)
  learner_model.eval()

  opponent_model = None
  if task["opponent_type"] == "model" and task["opponent_state_dict"] is not None:
    opponent_model = TransformerActorCritic()
    opponent_model.load_state_dict(task["opponent_state_dict"], strict=False)
    opponent_model.eval()

  # Roster & Environment Instantiation
  l_phs = [LocalPlaceholder() for _ in range(l_size)]
  o_phs = [LocalPlaceholder() for _ in range(o_size)]

  roster = []
  for i in range(l_size):
    roster.append(PlayerSlot(l_team, PlayerStats(f"L{i+1}", accel=3200.0), l_phs[i]))
  for j in range(o_size):
    roster.append(PlayerSlot(o_team, PlayerStats(f"O{j+1}", accel=3200.0), o_phs[j]))

  cfg = MatchConfig(
      mode=ClassicMatchMode(time_limit=task["max_steps"] / 60.0, score_limit=99),
      roster=roster,
      goal_height=task["goal_height"],
      pitch_width=task["pitch_width"],
      pitch_height=task["pitch_height"],
  )
  sim = Simulation(match_config=cfg, goal_height=task["goal_height"])
  if hasattr(sim, "mode"):
    sim.mode.state = "PLAYING"
    if hasattr(sim.mode, "score_limit"):
      sim.mode.score_limit = 999
    if hasattr(sim.mode, "reset_positions"):
      sim.mode.reset_positions = lambda *args, **kwargs: None

  _apply_eval_restart(sim, ep_idx=ep, is_initial=True)

  # Setup Heuristic Bot with Tactical Variety
  tactics = ["balanced", "defending", "attacking"]
  heuristic_bot = HeuristicBotController(
      TeamHeuristicCoordinator(team=o_team, strategy=tactics[ep % 3])
  )

  l_histories = [deque(maxlen=3) for _ in range(l_size)]
  o_histories = [deque(maxlen=3) for _ in range(o_size)]
  total_decisions = task["max_steps"] // task["action_repeat"]
  ep_reward = 0.0

  for _ in range(total_decisions):
    # 1. Learner Action Decision
    squad_l = sim.red_team if l_team == "red" else sim.blue_team
    l_egos, l_balls, l_mates, l_opps, l_masks = [], [], [], [], []
    for pl_idx, pl in enumerate(squad_l):
      toks = extract_entity_obs(sim, pl, l_team)
      dq = l_histories[pl_idx]
      if len(dq) == 0:
        for _ in range(3):
          dq.append(toks)
      else:
        dq.append(toks)

      l_egos.append(np.concatenate([f["ego"] for f in dq], axis=-1))
      l_balls.append(np.concatenate([f["ball"] for f in dq], axis=-1))
      l_mates.append(np.concatenate([f["teammates"] for f in dq], axis=-1))
      l_opps.append(np.concatenate([f["opponents"] for f in dq], axis=-1))
      l_masks.append(toks["key_padding_mask"])

    b_ego = torch.as_tensor(np.stack(l_egos), dtype=torch.float32)
    b_ball = torch.as_tensor(np.stack(l_balls), dtype=torch.float32)
    b_mates = torch.as_tensor(np.stack(l_mates), dtype=torch.float32)
    b_opps = torch.as_tensor(np.stack(l_opps), dtype=torch.float32)
    b_mask = torch.as_tensor(np.stack(l_masks), dtype=torch.bool)

    with torch.no_grad():
      m_logits, k_logits = learner_model.actor(b_ego, b_ball, b_mates, b_opps, b_mask)
      acts_l_m = torch.argmax(m_logits, dim=-1).tolist()
      acts_l_k = torch.argmax(k_logits, dim=-1).tolist()

    for pl_idx in range(l_size):
      ex, ey = _EGO_DIRS[acts_l_m[pl_idx]]
      k_val = bool(acts_l_k[pl_idx] == 1)
      l_phs[pl_idx].action = (Vec2(ex * sign, ey), k_val)

    # 2. Opponent Action Decision
    if task["opponent_type"] == "model" and opponent_model is not None:
      squad_o = sim.blue_team if l_team == "red" else sim.red_team
      o_egos, o_balls, o_mates, o_opps, o_masks = [], [], [], [], []
      for pl_idx, pl in enumerate(squad_o):
        toks = extract_entity_obs(sim, pl, o_team)
        dq = o_histories[pl_idx]
        if len(dq) == 0:
          for _ in range(3):
            dq.append(toks)
        else:
          dq.append(toks)

        o_egos.append(np.concatenate([f["ego"] for f in dq], axis=-1))
        o_balls.append(np.concatenate([f["ball"] for f in dq], axis=-1))
        o_mates.append(np.concatenate([f["teammates"] for f in dq], axis=-1))
        o_opps.append(np.concatenate([f["opponents"] for f in dq], axis=-1))
        o_masks.append(toks["key_padding_mask"])

      b_ego_o = torch.as_tensor(np.stack(o_egos), dtype=torch.float32)
      b_ball_o = torch.as_tensor(np.stack(o_balls), dtype=torch.float32)
      b_mates_o = torch.as_tensor(np.stack(o_mates), dtype=torch.float32)
      b_opps_o = torch.as_tensor(np.stack(o_opps), dtype=torch.float32)
      b_mask_o = torch.as_tensor(np.stack(o_masks), dtype=torch.bool)

      with torch.no_grad():
        m_logits_o, k_logits_o = opponent_model.actor(b_ego_o, b_ball_o, b_mates_o, b_opps_o, b_mask_o)
        acts_o_m = torch.argmax(m_logits_o, dim=-1).tolist()
        acts_o_k = torch.argmax(k_logits_o, dim=-1).tolist()

      for pl_idx in range(o_size):
        ex, ey = _EGO_DIRS[acts_o_m[pl_idx]]
        k_val_o = bool(acts_o_k[pl_idx] == 1)
        o_phs[pl_idx].action = (Vec2(ex * -sign, ey), k_val_o)

    elif task["opponent_type"] == "heuristic":
      squad_o = sim.blue_team if l_team == "red" else sim.red_team
      for pl_idx, opp_pl in enumerate(squad_o):
        g_idx = sim.all_players.index(opp_pl)
        o_phs[pl_idx].action = heuristic_bot.get_action(g_idx, sim)
    else:
      for pl_idx in range(o_size):
        dx, dy = random.choice(_EGO_DIRS)
        o_phs[pl_idx].action = (Vec2(dx, dy), random.random() < 0.20)

    # 3. Physics Steps
    for _ in range(task["action_repeat"]):
      goal = sim.step(1.0 / 60.0)
      if hasattr(sim, "mode"):
        sim.mode.state = "PLAYING"
        if hasattr(sim.mode, "score_limit"):
          sim.mode.score_limit = 999
      if goal is not None:
        scored = goal == f"{l_team}_goal"
        ep_reward += 1.0 if scored else -1.0
        _apply_eval_restart(sim, ep_idx=ep, is_initial=False)
        for dq in l_histories:
          dq.clear()
        for dq in o_histories:
          dq.clear()
        break

  scored_goals = sim.score_red if l_team == "red" else sim.score_blue
  conceded_goals = sim.score_blue if l_team == "red" else sim.score_red
  diff = scored_goals - conceded_goals

  if diff > 0:
    ep_reward += 1.0 + 0.1 * diff
    outcome = "win"
  elif diff < 0:
    ep_reward -= 1.0 + 0.1 * abs(diff)
    outcome = "loss"
  else:
    ep_reward -= 0.5
    outcome = "draw"

  return {
      "outcome": outcome,
      "scored": scored_goals,
      "conceded": conceded_goals,
      "reward": ep_reward,
  }


class PoolOpponentController(Controller):
  """Optimized opponent controller with in-memory checkpoint caching and fast inference."""

  _WEIGHTS_CACHE: dict[str, dict] = {}

  def __init__(
      self,
      pool_dir: str | None = None,
      team: str = "blue",
      device: str = "cpu",
      p_random: float = 0.05,
      p_heuristic: float = 0.50,
      frame_stack: int = 3,
  ):
    torch.set_num_threads(1)
    self.pool_dir = pool_dir
    self.team = team
    self.sign = 1.0 if team == "red" else -1.0
    self.device = torch.device(device)

    self.p_random = p_random
    self.p_heuristic = p_heuristic
    self.frame_stack = frame_stack
    self.history: dict[int, deque] = {}

    self.random_ctrl = RandomOpponentController()
    self.heuristic_ctrl = HeuristicBotController(
        TeamHeuristicCoordinator(team=team)
    )
    self.model = TransformerActorCritic().to(self.device)
    self.model.eval()

    self.current_mode = "random"

  def reset_opponent(self):
    self.history.clear()
    roll = random.random()

    if roll < self.p_random or not self.pool_dir or not os.path.exists(self.pool_dir):
      self.current_mode = "random"
      return

    if roll < (self.p_random + self.p_heuristic):
      self.current_mode = "heuristic"
      if hasattr(self.heuristic_ctrl, "coordinator") and hasattr(
          self.heuristic_ctrl.coordinator, "reset_strategy"
      ):
        self.heuristic_ctrl.coordinator.reset_strategy()
      return

    history_files = glob.glob(os.path.join(self.pool_dir, "history_*.pt"))
    latest_file = os.path.join(self.pool_dir, "latest.pt")
    target_file = None

    if os.path.exists(latest_file) and random.random() < 0.40:
      target_file = latest_file
    elif history_files:
      def _extract_step(filepath: str) -> int:
        base = os.path.basename(filepath)
        try:
          return int(base.replace("history_", "").replace(".pt", ""))
        except ValueError:
          return 0

      sorted_history = sorted(history_files, key=_extract_step)
      n_files = len(sorted_history)
      weights = np.exp(np.linspace(0.0, 2.5, num=n_files))
      weights /= weights.sum()
      target_file = random.choices(sorted_history, weights=weights, k=1)[0]
    elif os.path.exists(latest_file):
      target_file = latest_file

    if target_file:
      try:
        if target_file not in self._WEIGHTS_CACHE:
          ckpt = torch.load(target_file, map_location=self.device, weights_only=False)
          state_dict = (
              ckpt["model_state_dict"]
              if isinstance(ckpt, dict) and "model_state_dict" in ckpt
              else ckpt
          )
          self._WEIGHTS_CACHE[target_file] = state_dict
          if len(self._WEIGHTS_CACHE) > 12:
            oldest = next(iter(self._WEIGHTS_CACHE))
            del self._WEIGHTS_CACHE[oldest]

        self.model.load_state_dict(self._WEIGHTS_CACHE[target_file], strict=False)
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
    obs = extract_entity_obs(sim, player, self.team)

    if player_idx not in self.history or len(self.history[player_idx]) == 0:
      self.history[player_idx] = deque(maxlen=self.frame_stack)
      for _ in range(self.frame_stack):
        self.history[player_idx].append(obs)
    else:
      self.history[player_idx].append(obs)

    h_dq = self.history[player_idx]
    stacked_ego = np.concatenate([f["ego"] for f in h_dq], axis=-1)
    stacked_ball = np.concatenate([f["ball"] for f in h_dq], axis=-1)
    stacked_mates = np.concatenate([f["teammates"] for f in h_dq], axis=-1)
    stacked_opps = np.concatenate([f["opponents"] for f in h_dq], axis=-1)

    t_ego = torch.as_tensor(stacked_ego, dtype=torch.float32, device=self.device).unsqueeze(0)
    t_ball = torch.as_tensor(stacked_ball, dtype=torch.float32, device=self.device).unsqueeze(0)
    t_mates = torch.as_tensor(stacked_mates, dtype=torch.float32, device=self.device).unsqueeze(0)
    t_opps = torch.as_tensor(stacked_opps, dtype=torch.float32, device=self.device).unsqueeze(0)
    t_mask = torch.as_tensor(obs["key_padding_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)

    m_idx, kick = self.model.actor.predict_action(t_ego, t_ball, t_mates, t_opps, t_mask)
    ego_x, ego_y = _EGO_DIRS[m_idx]
    return Vec2(ego_x * self.sign, ego_y), kick


class SelfPlayPool:
  """Manages model checkpoints and high-throughput multi-core batched evaluations."""

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
      num_episodes: int = 36,
      team_size: int = 3,
      opp_team_size: int | None = None,
      goal_height: float | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      max_steps: int = 3600,
      action_repeat: int = 10,
      eval_seed: int = 42,
      pool: mp.Pool | None = None,
  ) -> dict:
    effective_opp_size = opp_team_size if opp_team_size is not None else team_size

    # Move weights to CPU dictionaries before inter-process serialization
    l_state = {k: v.detach().cpu() for k, v in learner_model.state_dict().items()}
    o_state = None
    if opponent_model is not None:
      o_state = {k: v.detach().cpu() for k, v in opponent_model.state_dict().items()}

    tasks = [
        {
            "ep": ep,
            "opponent_type": opponent_type,
            "learner_state_dict": l_state,
            "opponent_state_dict": o_state,
            "team_size": team_size,
            "opp_team_size": effective_opp_size,
            "goal_height": goal_height,
            "pitch_width": pitch_width,
            "pitch_height": pitch_height,
            "max_steps": max_steps,
            "action_repeat": action_repeat,
            "eval_seed": eval_seed,
        }
        for ep in range(num_episodes)
    ]

    # Run across multiprocessing pool if available; otherwise run sequentially
    if pool is not None:
      results = pool.map(_eval_single_episode_worker, tasks)
    else:
      results = [_eval_single_episode_worker(t) for t in tasks]

    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    draws = sum(1 for r in results if r["outcome"] == "draw")
    total_scored = sum(r["scored"] for r in results)
    total_conceded = sum(r["conceded"] for r in results)
    total_rewards = [r["reward"] for r in results]

    mean_reward = float(np.mean(total_rewards))
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
      num_episodes: int | dict[str, int] = 36,
      tier_ratios: dict[str, float] | None = None,
      action_repeat: int = 10,
      max_steps: int = 3600,
      device: torch.device = torch.device("cpu"),
  ) -> tuple[bool, dict, tuple]:
    results = {}
    filters = filter_thresholds or {}
    effective_opp_size = opp_team_size if opp_team_size is not None else team_size

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

    # Distribute parallel tasks across half of available host CPU cores
    total_cpus = os.cpu_count() or 4
    n_workers = max(1, total_cpus // 2)
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=n_workers) as pool:
      # Tier 1: Random Bot
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
            num_episodes=eps,
            max_steps=max_steps,
            action_repeat=action_repeat,
            pool=pool,
        )
        if "random" in filters and results["random"]["win_rate"] < filters["random"]:
          return False, results, self.best_score

      # Tier 2: Heuristic Bot
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
            num_episodes=eps,
            max_steps=max_steps,
            action_repeat=action_repeat,
            pool=pool,
        )
        if "heuristic" in filters and results["heuristic"]["win_rate"] < filters["heuristic"]:
          return False, results, self.best_score

      # Tier 3: Champion (Self-Play Model)
      if "champion" in active_tiers:
        eps = tier_ep_counts.get("champion", 30)
        if os.path.exists(self.champion_path):
          champ = TransformerActorCritic().to("cpu")
          ckpt = torch.load(self.champion_path, map_location="cpu", weights_only=False)
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
              num_episodes=eps,
              max_steps=max_steps,
              action_repeat=action_repeat,
              pool=pool,
          )
        else:
          results["champion"] = {
              "win_rate": 1.0,
              "mean_reward": 1.0,
              "net": 1,
              "scored": 1,
              "score_tuple": (1.0, 1, 1.0, 1),
          }

        if "champion" in filters and results["champion"]["win_rate"] < filters["champion"]:
          return False, results, self.best_score

    cand = results[target_tier]

    # Priority: 1. Win Rate -> 2. Net Goals -> 3. Mean Reward -> 4. Goals Scored
    cand_score = (
        cand["win_rate"],
        cand["net"],
        cand["mean_reward"],
        cand["scored"],
    )

    if target_tier == "champion":
      # Self-play promotion benchmark: must beat champion with positive net diff
      is_promoted = cand["win_rate"] >= 0.30 and cand["net"] >= 5
      return is_promoted, results, cand_score

    # Strict lexicographical priority comparison
    if cand_score > self.best_score:
      prev_score = self.best_score
      self.best_score = cand_score
      return True, results, prev_score

    return False, results, self.best_score