import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from config.physics_config import PhysicsConfig
from src.engine.controllers import Controller
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_2.obs import (
    ACTOR_OBS_DIM,
    CRITIC_STATE_DIM,
    extract_actor_obs,
    extract_global_state,
)


class ActionPlaceholder(Controller):
  """Holds discrete action outputs injected directly by PPO rollouts."""

  def __init__(self):
    self.action = (Vec2(0.0, 0.0), False)

  def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
    return self.action


class RandomOpponentController(Controller):
  """Zero-dependency uniform random baseline."""

  def __init__(self):
    self._dirs = [
        Vec2(0, 0),
        Vec2(0, -1),
        Vec2(0, 1),
        Vec2(-1, 0),
        Vec2(1, 0),
        Vec2(-1, -1),
        Vec2(1, -1),
        Vec2(-1, 1),
        Vec2(1, 1),
    ]

  def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
    return random.choice(self._dirs), random.random() < 0.20


class MatchEnv(gym.Env):
  """Continuous Fixed-Time Match Environment with Point-Symmetric Resets."""

  metadata = {"render_modes": []}

  def __init__(
      self,
      team_size: int = 2,
      learner_team_size: int | None = None,
      opp_team_size: int | None = None,
      learner_team: str = "red",
      max_round_steps: int = 2700,
      goal_height: float | None = None,
      opponent_controller: Controller | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
  ):
    super().__init__()
    # Decouple team sizes while preserving team_size for MAPPO buffer sizing
    self.learner_team_size = learner_team_size or team_size
    self.opp_team_size = opp_team_size or team_size
    self.team_size = self.learner_team_size

    self.learner_team = learner_team
    self.opp_team = "blue" if learner_team == "red" else "red"
    self.max_round_steps = max_round_steps
    self.goal_height = goal_height
    self.pitch_width = pitch_width
    self.pitch_height = pitch_height
    self.opponent_controller = opponent_controller or RandomOpponentController()

    self._ego_dirs = [
        (0.0, 0.0),  # 0: None
        (0.0, -1.0),  # 1: Up
        (0.0, 1.0),  # 2: Down
        (-1.0, 0.0),  # 3: Backward
        (1.0, 0.0),  # 4: Forward
        (-1.0, -1.0),  # 5: Backward-Up
        (1.0, -1.0),  # 6: Forward-Up
        (-1.0, 1.0),  # 7: Backward-Down
        (1.0, 1.0),  # 8: Forward-Down
    ]

    # Spaces strictly track the learner squad dimensions
    if self.learner_team_size == 1:
      self.observation_space = spaces.Dict({
          "obs": spaces.Box(
              -1.0, 1.0, shape=(ACTOR_OBS_DIM,), dtype=np.float32
          ),
          "state": spaces.Box(
              -1.0, 1.0, shape=(CRITIC_STATE_DIM,), dtype=np.float32
          ),
      })
      self.action_space = spaces.MultiDiscrete([9, 2])
    else:
      self.observation_space = spaces.Dict({
          "obs": spaces.Box(
              -1.0,
              1.0,
              shape=(self.learner_team_size, ACTOR_OBS_DIM),
              dtype=np.float32,
          ),
          "state": spaces.Box(
              -1.0, 1.0, shape=(CRITIC_STATE_DIM,), dtype=np.float32
          ),
      })
      self.action_space = spaces.MultiDiscrete(
          [[9, 2]] * self.learner_team_size
      )

    self.current_step = 0
    self.match_time_remaining = float(self.max_round_steps) / 60.0
    self.sim: Simulation | None = None
    self.learner_slots: list[tuple[int, PlayerSlot]] = []
    self.opp_slots: list[tuple[int, PlayerSlot]] = []
    self._init_simulation()


  def _init_simulation(self):
    roster = []
    self.learner_slots = []
    self.opp_slots = []

    # 1. Populate Learner Squad (e.g., 2 RL players)
    for i in range(self.learner_team_size):
      ph = ActionPlaceholder()
      slot = PlayerSlot(
          self.learner_team,
          PlayerStats(f"Learner_{i+1}", accel=3200.0),
          ph,
      )
      roster.append(slot)
      self.learner_slots.append((i, slot))

    # 2. Populate Opponent Squad (e.g., 3 Heuristic bots)
    for j in range(self.opp_team_size):
      ph_opp = ActionPlaceholder()
      slot = PlayerSlot(
          self.opp_team,
          PlayerStats(f"Opponent_{j+1}", accel=3200.0),
          ph_opp,
      )
      roster.append(slot)
      self.opp_slots.append((self.learner_team_size + j, slot))

    match_cfg = MatchConfig(
        mode=ClassicMatchMode(
            time_limit=self.max_round_steps / 60.0, score_limit=99
        ),
        roster=roster,
        pitch_width=self.pitch_width,
        pitch_height=self.pitch_height,
        goal_height=self.goal_height,
    )
    self.sim = Simulation(
        center_x=self.pitch_width / 2.0,
        center_y=self.pitch_height / 2.0,
        match_config=match_cfg,
        goal_height=self.goal_height,
    )

    # ── DISABLE INTERNAL ENGINE CELEBRATION & TELEPORTS ──
    if hasattr(self.sim, "mode"):
      self.sim.mode.state = "PLAYING"
      if hasattr(self.sim.mode, "reset_positions"):
        self.sim.mode.reset_positions = lambda *args, **kwargs: None


  def _reset_pitch_state(self, standard_kickoff: bool = False):
    """Executes 6 structured tactical phase resets to eliminate clustering."""
    p = self.sim.pitch
    ball = self.sim.ball
    roll = random.random()

    # ── 1. Standard Center Kickoff (10%) ──
    if standard_kickoff or roll < 0.10:
      ball.pos = Vec2(self.sim.center.x, self.sim.center.y)
      ball.vel = Vec2(0.0, 0.0)
      r_coords = [
          Vec2(self.sim.center.x - 140.0, self.sim.center.y),
          Vec2(self.sim.center.x - 240.0, self.sim.center.y + 110.0),
      ]
      b_coords = [
          Vec2(self.sim.center.x + 140.0, self.sim.center.y),
          Vec2(self.sim.center.x + 240.0, self.sim.center.y - 110.0),
          Vec2(p.right - 100.0, self.sim.center.y),  # Extra defender/GK if 3v3
      ]
      self._apply_tactical_layout(r_coords, b_coords)
      return

    # ── 2. Defensive Breakout & Clearance (20%) ──
    # Red P1 trapped in box corner with ball; Red P2 open in midfield
    elif roll < 0.30:
      bx = p.left + 120.0
      by = random.choice([p.top + 90.0, p.bottom - 90.0])
      ball.pos = Vec2(bx, by)
      ball.vel = Vec2(0.0, 0.0)

      outlet_y = p.bottom - 130.0 if by < self.sim.center.y else p.top + 130.0
      r_coords = [
          Vec2(bx - 35.0, by),  # Trapped press target
          Vec2(self.sim.center.x - 80.0, outlet_y),  # Wide outlet runner
      ]
      b_coords = [
          Vec2(bx + 65.0, by),  # Pressing striker
          Vec2(self.sim.center.x + 80.0, self.sim.center.y),  # Covering anchor
          Vec2(p.right - 60.0, self.sim.center.y),  # Deep GK
      ]
      self._apply_tactical_layout(r_coords, b_coords)
      return

    # ── 3. Wing Cross & Cutback (20%) ──
    # Red P1 drives down the wing; Red P2 runs central to attack the box
    elif roll < 0.50:
      wing_y = random.choice([p.top + 60.0, p.bottom - 60.0])
      bx = p.right - 170.0
      ball.pos = Vec2(bx, wing_y)
      ball.vel = Vec2(0.0, 0.0)

      r_coords = [
          Vec2(bx - 35.0, wing_y),  # Winger
          Vec2(p.right - 200.0, self.sim.center.y),  # Target man waiting central
      ]
      b_coords = [
          Vec2(p.right - 55.0, self.sim.center.y),  # Goalkeeper on line
          Vec2(p.right - 140.0, wing_y + (60.0 if wing_y < self.sim.center.y else -60.0)),  # Wing defender
          Vec2(p.right - 160.0, self.sim.center.y),  # Central sweeper
      ]
      self._apply_tactical_layout(r_coords, b_coords)
      return

    # ── 4. 2v1 Fast-Break Overload (20%) ──
    # Red breaks with forward momentum in two lateral lanes
    elif roll < 0.70:
      bx = self.sim.center.x - 40.0
      by = self.sim.center.y - 120.0
      ball.pos = Vec2(bx, by)
      ball.vel = Vec2(140.0, 0.0)

      r_coords = [
          Vec2(bx - 35.0, by),  # Ball carrier (top lane)
          Vec2(bx - 35.0, self.sim.center.y + 130.0),  # Weak-side runner (bottom lane)
      ]
      b_coords = [
          Vec2(p.right - 220.0, self.sim.center.y),  # Back-pedaling defender
          Vec2(p.right - 60.0, self.sim.center.y),  # Goalkeeper
          Vec2(p.right - 180.0, self.sim.center.y + 120.0),  # Recovery runner
      ]
      self._apply_tactical_layout(r_coords, b_coords)
      return

    # ── 5. Emergency Goal-Line Stand (15%) ──
    # Blue takes a heavy shot; Red must defend the goal and clear
    elif roll < 0.85:
      bx = p.left + 220.0
      by = random.uniform(p.top + 160.0, p.bottom - 160.0)
      ball.pos = Vec2(bx, by)
      ball.vel = Vec2(-180.0, 0.0)

      r_coords = [
          Vec2(p.left + 50.0, self.sim.center.y),  # Red GK
          Vec2(p.left + 140.0, by + 40.0),  # Red Sweeper
      ]
      b_coords = [
          Vec2(bx + 45.0, by),  # Blue Striker
          Vec2(self.sim.center.x, self.sim.center.y),  # Blue Midfielder
          Vec2(p.right - 70.0, self.sim.center.y),  # Blue GK
      ]
      self._apply_tactical_layout(r_coords, b_coords)
      return

    # ── 6. Staggered Midfield Scramble (15%) ──
    else:
      bx = random.uniform(self.sim.center.x - 80.0, self.sim.center.x + 80.0)
      by = random.uniform(p.top + 120.0, p.bottom - 120.0)
      ball.pos = Vec2(bx, by)
      ball.vel = Vec2(random.uniform(-70.0, 70.0), random.uniform(-70.0, 70.0))

      r_coords = [
          Vec2(bx - 120.0, by),
          Vec2(bx - 260.0, p.bottom - 140.0 if by < self.sim.center.y else p.top + 140.0),
      ]
      b_coords = [
          Vec2(bx + 120.0, by),
          Vec2(bx + 260.0, p.top + 140.0 if by < self.sim.center.y else p.bottom - 140.0),
          Vec2(p.right - 80.0, self.sim.center.y),
      ]
      self._apply_tactical_layout(r_coords, b_coords)

  def _apply_tactical_layout(self, r_coords: list[Vec2], b_coords: list[Vec2]):
    """Safely clamps positions to pitch boundaries and resets all player velocities."""
    p = self.sim.pitch
    safe_m = 48.0

    for idx, pl in enumerate(self.sim.red_team):
      target = (
          r_coords[idx]
          if idx < len(r_coords)
          else Vec2(self.sim.center.x - 180.0 - idx * 40.0, self.sim.center.y)
      )
      pl.pos = Vec2(
          max(p.left + safe_m, min(p.right - safe_m, target.x)),
          max(p.top + safe_m, min(p.bottom - safe_m, target.y)),
      )
      pl.vel = Vec2(0.0, 0.0)
      pl.kick_cooldown_timer = 0.0

    for idx, pl in enumerate(self.sim.blue_team):
      target = (
          b_coords[idx]
          if idx < len(b_coords)
          else Vec2(self.sim.center.x + 180.0 + idx * 40.0, self.sim.center.y)
      )
      pl.pos = Vec2(
          max(p.left + safe_m, min(p.right - safe_m, target.x)),
          max(p.top + safe_m, min(p.bottom - safe_m, target.y)),
      )
      pl.vel = Vec2(0.0, 0.0)
      pl.kick_cooldown_timer = 0.0

  def _apply_squad_positions(self, rx: float, ry: float, bx: float, by: float):
    """Universal NvN squad placer with point-symmetric randomization.

    Supports 1v1, 2v2, 3v3, etc. without hand-crafted roles or formation bias.
    """
    p = self.sim.pitch
    ball = self.sim.ball
    center = self.sim.center

    safe_m = 48.0
    min_player_dist = 52.0  # Prevents disc-overlap collisions (radius ~15px)
    min_ball_dist = 36.0  # Prevents spawning inside the ball (radius ~10px)

    # 1. Place Primary Duelists (Index 0)
    self.sim.red_team[0].pos = Vec2(rx, ry)
    self.sim.red_team[0].vel = Vec2(0.0, 0.0)
    self.sim.red_team[0].kick_cooldown_timer = 0.0

    self.sim.blue_team[0].pos = Vec2(bx, by)
    self.sim.blue_team[0].vel = Vec2(0.0, 0.0)
    self.sim.blue_team[0].kick_cooldown_timer = 0.0

    placed_red = [self.sim.red_team[0].pos]
    placed_blue = [self.sim.blue_team[0].pos]

    team_size = len(self.sim.red_team)
    if team_size <= 1:
      return

    # 2. Randomly Place Additional Teammates (Indices 1 .. N-1) with Strict Fairness
    for idx in range(1, team_size):
      placed = False

      for _ in range(60):
        # 50% Ball-Symmetric (equal distance to ball), 50% Pitch-Symmetric (equal goal/pitch coverage)
        if random.random() < 0.50:
          # Ball-Centric: random radial offset from ball in any direction
          dist = random.uniform(70.0, min(520.0, p.width * 0.45))
          angle = random.uniform(-math.pi, math.pi)
          vx = dist * math.cos(angle)
          vy = dist * math.sin(angle)

          r_pos = Vec2(ball.pos.x - vx, ball.pos.y - vy)
          b_pos = Vec2(ball.pos.x + vx, ball.pos.y + vy)
        else:
          # Pitch-Centric: point reflection across field center (2*C - P)
          rx_cand = random.uniform(
              p.left + safe_m, center.x + (p.width * 0.15)
          )
          ry_cand = random.uniform(p.top + safe_m, p.bottom - safe_m)

          r_pos = Vec2(rx_cand, ry_cand)
          b_pos = Vec2(2.0 * center.x - rx_cand, 2.0 * center.y - ry_cand)

        # A. Pitch Boundary Checks
        if not (
            p.left + safe_m <= r_pos.x <= p.right - safe_m
            and p.top + safe_m <= r_pos.y <= p.bottom - safe_m
            and p.left + safe_m <= b_pos.x <= p.right - safe_m
            and p.top + safe_m <= b_pos.y <= p.bottom - safe_m
        ):
          continue

        # B. Ball Clearance Checks
        if (
            r_pos.distance_to(ball.pos) < min_ball_dist
            or b_pos.distance_to(ball.pos) < min_ball_dist
        ):
          continue

        # C. Inter-Player Collision Checks
        collision = False
        for existing in placed_red + placed_blue:
          if (
              r_pos.distance_to(existing) < min_player_dist
              or b_pos.distance_to(existing) < min_player_dist
          ):
            collision = True
            break

        if r_pos.distance_to(b_pos) < min_player_dist:
          collision = True

        if collision:
          continue

        # Valid non-overlapping symmetric spawn found
        self.sim.red_team[idx].pos = r_pos
        self.sim.red_team[idx].vel = Vec2(0.0, 0.0)
        self.sim.red_team[idx].kick_cooldown_timer = 0.0

        self.sim.blue_team[idx].pos = b_pos
        self.sim.blue_team[idx].vel = Vec2(0.0, 0.0)
        self.sim.blue_team[idx].kick_cooldown_timer = 0.0

        placed_red.append(r_pos)
        placed_blue.append(b_pos)
        placed = True
        break

      # Deterministic fallback if random sampling gets crowded
      if not placed:
        fallback_offset = 130.0 * idx
        fallback_r = Vec2(
            max(p.left + safe_m, rx - fallback_offset),
            min(
                p.bottom - safe_m,
                max(p.top + safe_m, ry + (70.0 if idx % 2 == 1 else -70.0)),
            ),
        )
        fallback_b = Vec2(
            min(p.right - safe_m, bx + fallback_offset),
            min(
                p.bottom - safe_m,
                max(p.top + safe_m, by - (70.0 if idx % 2 == 1 else -70.0)),
            ),
        )

        self.sim.red_team[idx].pos = fallback_r
        self.sim.red_team[idx].vel = Vec2(0.0, 0.0)
        self.sim.red_team[idx].kick_cooldown_timer = 0.0

        self.sim.blue_team[idx].pos = fallback_b
        self.sim.blue_team[idx].vel = Vec2(0.0, 0.0)
        self.sim.blue_team[idx].kick_cooldown_timer = 0.0

        placed_red.append(fallback_r)
        placed_blue.append(fallback_b)


  def _get_obs_payload(self) -> dict[str, np.ndarray]:
    team_squad = (
        self.sim.red_team
        if self.learner_team == "red"
        else self.sim.blue_team
    )
    obs_list = [
        extract_actor_obs(self.sim, player, self.learner_team)
        for player in team_squad
    ]
    state = extract_global_state(self.sim, self.learner_team)

    if self.learner_team_size == 1:
      return {"obs": obs_list[0], "state": state}
    return {"obs": np.array(obs_list, dtype=np.float32), "state": state}

  def reset(self, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    if seed is not None:
      random.seed(seed)
      np.random.seed(seed)

    self.current_step = 0
    self.match_time_remaining = float(self.max_round_steps) / 60.0
    self.sim.score_red = 0
    self.sim.score_blue = 0

    if hasattr(self.opponent_controller, "reset_opponent"):
      self.opponent_controller.reset_opponent()

    if hasattr(self.sim.mode, "time_remaining"):
      self.sim.mode.time_remaining = self.match_time_remaining
      self.sim.mode.state = "PLAYING"

    is_standard = random.random() < 0.20
    self._reset_pitch_state(standard_kickoff=is_standard)
    return self._get_obs_payload(), {}

  def step(self, action):
    dt = 1.0 / 60.0
    action_repeat = 4

    # 1. Update Learner Actions (15 Hz decision window)
    sign = 1.0 if self.learner_team == "red" else -1.0
    actions = [action] if self.learner_team_size == 1 else action
    for i, (_, slot) in enumerate(self.learner_slots):
      m_idx = int(actions[i][0])
      kick = bool(actions[i][1])
      ego_x, ego_y = self._ego_dirs[m_idx]
      slot.controller.action = (Vec2(ego_x * sign, ego_y), kick)

    # 2. Update Opponent Actions (15 Hz synchronized)
    for j, slot in self.opp_slots:
      slot.controller.action = self.opponent_controller.get_action(j, self.sim)

    total_reward = 0.0

    # 3. Physics Substeps (Continuous Match Loop)
    for _ in range(action_repeat):
      self.current_step += 1
      self.match_time_remaining = max(0.0, self.match_time_remaining - dt)

      if hasattr(self.sim.mode, "time_remaining"):
        self.sim.mode.time_remaining = self.match_time_remaining

      goal_event = self.sim.step(dt)

      if hasattr(self.sim, "mode"):
        self.sim.mode.state = "PLAYING"

      # In-Match Continuous Goal Handling
      if goal_event is not None:
        scored = goal_event == f"{self.learner_team}_goal"
        total_reward += 1.0 if scored else -1.0

        # Restart immediately via 20/80 setup without terminating match
        is_standard = random.random() < 0.20
        self._reset_pitch_state(standard_kickoff=is_standard)
        break

      if (
          self.current_step >= self.max_round_steps
          or self.match_time_remaining <= 0.0
      ):
        break

    # 4. Termination & Final Whistle Score Differential Bonus
    terminated = False
    truncated = (
        self.current_step >= self.max_round_steps
        or self.match_time_remaining <= 0.0
    )

    if truncated:
      my_score = (
          self.sim.score_red
          if self.learner_team == "red"
          else self.sim.score_blue
      )
      opp_score = (
          self.sim.score_blue
          if self.learner_team == "red"
          else self.sim.score_red
      )
      score_diff = my_score - opp_score

      if score_diff > 0:
        total_reward += 1.0 + 0.1 * score_diff
      elif score_diff < 0:
        total_reward -= 1.0 + 0.1 * abs(score_diff)
      else:
        total_reward -= 0.1

    info = {
        "score_red": self.sim.score_red,
        "score_blue": self.sim.score_blue,
        "match_time_remaining": self.match_time_remaining,
    }

    return self._get_obs_payload(), float(total_reward), terminated, truncated, info