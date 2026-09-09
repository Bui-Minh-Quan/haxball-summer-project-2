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
      team_size: int = 1,
      learner_team: str = "red",
      max_round_steps: int = 1800,  # 30s default continuous match @ 60Hz
      goal_height: float | None = None,
      opponent_controller: Controller | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
  ):
    super().__init__()
    self.team_size = team_size
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

    if self.team_size == 1:
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
              shape=(self.team_size, ACTOR_OBS_DIM),
              dtype=np.float32,
          ),
          "state": spaces.Box(
              -1.0, 1.0, shape=(CRITIC_STATE_DIM,), dtype=np.float32
          ),
      })
      self.action_space = spaces.MultiDiscrete([[9, 2]] * self.team_size)

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

    for i in range(self.team_size):
      ph = ActionPlaceholder()
      slot = PlayerSlot(
          self.learner_team,
          PlayerStats(f"Learner_{i+1}", accel=3200.0),
          ph,
      )
      roster.append(slot)
      self.learner_slots.append((i, slot))

    for j in range(self.team_size):
      ph_opp = ActionPlaceholder()
      slot = PlayerSlot(
          self.opp_team,
          PlayerStats(f"Opponent_{j+1}", accel=3200.0),
          ph_opp,
      )
      roster.append(slot)
      self.opp_slots.append((j, slot))

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

  def _reset_pitch_state(self, standard_kickoff: bool = False):
    """Executes fair 20/80 kickoffs dynamically adapted to pitch and goal geometry."""
    p = self.sim.pitch
    ball = self.sim.ball

    pitch_w = p.right - p.left
    pitch_h = p.bottom - p.top
    is_wide_goal = bool(self.goal_height and self.goal_height >= 400.0)

    # ── 1. Standard Center Kickoff (20% or Fallback) ──
    if standard_kickoff:
      ball.pos = Vec2(self.sim.center.x, self.sim.center.y)
      ball.vel = Vec2(0.0, 0.0)

      offset_x = min(140.0, pitch_w * 0.16)
      for idx, pl in enumerate(self.sim.red_team):
        pl.pos = Vec2(
            self.sim.center.x - offset_x, self.sim.center.y + (idx * 50.0)
        )
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0

      for idx, pl in enumerate(self.sim.blue_team):
        pl.pos = Vec2(
            self.sim.center.x + offset_x, self.sim.center.y - (idx * 50.0)
        )
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0
      return

    # ── 2. Phase 1: Wide-Net Striking Discovery (Clear Firing Lane) ──
    if is_wide_goal:
      # Ball spawns in the central area
      bx = random.uniform(p.left + 200.0, p.right - 220.0)
      by = random.uniform(p.top + 70.0, p.bottom - 70.0)
      ball.pos = Vec2(bx, by)
      ball.vel = Vec2(0.0, 0.0)

      # Learner (Red) spawns behind the ball with forward orientation
      dist_l = random.uniform(80.0, 150.0)
      angle_l = random.uniform(-math.pi / 4, math.pi / 4)
      rx = max(p.left + 45.0, bx - dist_l * math.cos(angle_l))
      ry = min(
          max(p.top + 45.0, by - dist_l * math.sin(angle_l)), p.bottom - 45.0
      )

      # Opponent (Blue) spawns deep or on the flanks, clearing the direct shooting lane
      opp_x = random.uniform(p.right - 180.0, p.right - 60.0)
      opp_y = random.choice([
          random.uniform(p.top + 50.0, p.top + 120.0),
          random.uniform(p.bottom - 120.0, p.bottom - 50.0),
      ])

      for idx, pl in enumerate(self.sim.red_team):
        pl.pos = Vec2(rx, ry + (idx * 40.0))
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0

      for idx, pl in enumerate(self.sim.blue_team):
        pl.pos = Vec2(opp_x, opp_y - (idx * 40.0))
        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0
      return

    # ── 3. Phase 2 & 3: Regulation Equidistant Point-Symmetric Contest ──
    margin_x = min(240.0, pitch_w * 0.22)
    margin_y = min(110.0, pitch_h * 0.18)
    bx = random.uniform(p.left + margin_x, p.right - margin_x)
    by = random.uniform(p.top + margin_y, p.bottom - margin_y)
    ball.pos = Vec2(bx, by)
    ball.vel = Vec2(0.0, 0.0)

    safe_m = 45.0
    placed = False
    max_dist = min(220.0, pitch_w * 0.20)

    for _ in range(40):
      dist = random.uniform(150.0, max_dist)
      # Strictly forward angles: cos(angle) > 0 guarantees Red is to the left of the ball
      angle = random.uniform(-math.pi / 4, math.pi / 4)
      vx = dist * math.cos(angle)
      vy = dist * math.sin(angle)

      r_x, r_y = bx - vx, by - vy
      b_x, b_y = bx + vx, by + vy

      if (
          p.left + safe_m <= r_x <= p.right - safe_m
          and p.top + safe_m <= r_y <= p.bottom - safe_m
          and p.left + safe_m <= b_x <= p.right - safe_m
          and p.top + safe_m <= b_y <= p.bottom - safe_m
      ):
        for idx, pl in enumerate(self.sim.red_team):
          pl.pos = Vec2(r_x, r_y + (idx * 45.0))
          pl.vel = Vec2(0.0, 0.0)
          pl.kick_cooldown_timer = 0.0

        for idx, pl in enumerate(self.sim.blue_team):
          pl.pos = Vec2(b_x, b_y - (idx * 45.0))
          pl.vel = Vec2(0.0, 0.0)
          pl.kick_cooldown_timer = 0.0

        placed = True
        break

    if not placed:
      self._reset_pitch_state(standard_kickoff=True)
      
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

    if self.team_size == 1:
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
    actions = [action] if self.team_size == 1 else action
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

      # In-Match Continuous Goal Handling
      if goal_event is not None:
        scored = goal_event == f"{self.learner_team}_goal"
        if goal_event == "red_goal":
          self.sim.score_red += 1
        elif goal_event == "blue_goal":
          self.sim.score_blue += 1

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