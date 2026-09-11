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
  """Continuous Fixed-Time Match Environment with Standard Half-Pitch Kickoff Resets."""

  metadata = {"render_modes": []}

  def __init__(
      self,
      team_size: int = 2,
      learner_team_size: int | None = None,
      opp_team_size: int | None = None,
      learner_team: str = "red",
      max_round_steps: int = 3600,  # 60s at 60 Hz physics
      action_repeat: int = 10,
      goal_height: float | None = None,
      opponent_controller: Controller | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      heuristic_accel: float = 3200.0,  
      heuristic_kick: float = 1200.0,
  ):
    super().__init__()
    self.learner_team_size = learner_team_size or team_size
    self.opp_team_size = opp_team_size or team_size
    self.team_size = self.learner_team_size

    self.learner_team = learner_team
    self.opp_team = "blue" if learner_team == "red" else "red"
    self.max_round_steps = max_round_steps
    self.action_repeat = action_repeat
    self.goal_height = goal_height
    self.pitch_width = pitch_width
    self.pitch_height = pitch_height
    self.opponent_controller = opponent_controller or RandomOpponentController()
    self.heuristic_accel = heuristic_accel
    self.heuristic_kick = heuristic_kick

    self._ego_dirs = [
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

    # 1. Populate Learner Squad
    for i in range(self.learner_team_size):
      ph = ActionPlaceholder()
      slot = PlayerSlot(
          self.learner_team,
          PlayerStats(f"Learner_{i+1}", accel=3200.0),
          ph,
      )
      roster.append(slot)
      self.learner_slots.append((i, slot))

    # 2. Populate Opponent Squad
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

    if hasattr(self.sim, "mode"):
      self.sim.mode.state = "PLAYING"
      if hasattr(self.sim.mode, "reset_positions"):
        self.sim.mode.reset_positions = lambda *args, **kwargs: None

  def _reset_kickoff(self):
    """Clean standard kickoff: ball dead center, players spread across their own half."""
    p = self.sim.pitch
    c = self.sim.center
    safe_m = 50.0
    min_player_dist = 52.0

    # 1. Ball dead-center with zero velocity
    self.sim.ball.pos = Vec2(c.x, c.y)
    self.sim.ball.vel = Vec2(0.0, 0.0)

    placed_positions: list[Vec2] = []

    # 2. Red Team in Left Half (x < center.x)
    for idx, red_player in enumerate(self.sim.red_team):
      placed = False
      for _ in range(50):
        cand = Vec2(
            random.uniform(p.left + safe_m, c.x - 40.0),
            random.uniform(p.top + safe_m, p.bottom - safe_m),
        )
        if all(cand.distance_to(pos) >= min_player_dist for pos in placed_positions):
          red_player.pos = cand
          placed_positions.append(cand)
          placed = True
          break

      if not placed:
        fallback_x = c.x - 100.0 - (idx * 80.0)
        fallback_y = c.y + (70.0 if idx % 2 == 1 else -70.0)
        red_player.pos = Vec2(max(p.left + safe_m, fallback_x), fallback_y)
        placed_positions.append(red_player.pos)

      red_player.vel = Vec2(0.0, 0.0)
      red_player.kick_cooldown_timer = 0.0

    # 3. Blue Team in Right Half (x > center.x)
    for idx, blue_player in enumerate(self.sim.blue_team):
      placed = False
      for _ in range(50):
        cand = Vec2(
            random.uniform(c.x + 40.0, p.right - safe_m),
            random.uniform(p.top + safe_m, p.bottom - safe_m),
        )
        if all(cand.distance_to(pos) >= min_player_dist for pos in placed_positions):
          blue_player.pos = cand
          placed_positions.append(cand)
          placed = True
          break

      if not placed:
        fallback_x = c.x + 100.0 + (idx * 80.0)
        fallback_y = c.y + (70.0 if idx % 2 == 1 else -70.0)
        blue_player.pos = Vec2(min(p.right - safe_m, fallback_x), fallback_y)
        placed_positions.append(blue_player.pos)

      blue_player.vel = Vec2(0.0, 0.0)
      blue_player.kick_cooldown_timer = 0.0

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

    # Dynamic Stat Buffing: Heuristic gets buffed, Self-Play stays standard (3200 / 1200)
    mode = getattr(self.opponent_controller, "current_mode", "heuristic")
    is_heuristic = (mode == "heuristic")
    target_accel = self.heuristic_accel if is_heuristic else 3200.0
    target_kick = self.heuristic_kick if is_heuristic else 1200.0

    opp_team = self.sim.blue_team if self.learner_team == "red" else self.sim.red_team
    for player in opp_team:
      player.stats.accel = target_accel
      player.stats.kick_strength = target_kick

    if hasattr(self.sim.mode, "time_remaining"):
      self.sim.mode.time_remaining = self.match_time_remaining
      self.sim.mode.state = "PLAYING"

    self._reset_kickoff()
    return self._get_obs_payload(), {}

  def step(self, action):
    dt = 1.0 / 60.0
    action_repeat = self.action_repeat

    # 1. Update Learner Actions (15 Hz)
    sign = 1.0 if self.learner_team == "red" else -1.0
    actions = [action] if self.learner_team_size == 1 else action
    for i, (_, slot) in enumerate(self.learner_slots):
      m_idx = int(actions[i][0])
      kick = bool(actions[i][1])
      ego_x, ego_y = self._ego_dirs[m_idx]
      slot.controller.action = (Vec2(ego_x * sign, ego_y), kick)

    # 2. Update Opponent Actions (15 Hz)
    for j, slot in self.opp_slots:
      slot.controller.action = self.opponent_controller.get_action(j, self.sim)

    total_reward = 0.0

    # 3. Physics Substeps
    for _ in range(action_repeat):
      self.current_step += 1
      self.match_time_remaining = max(0.0, self.match_time_remaining - dt)

      if hasattr(self.sim.mode, "time_remaining"):
        self.sim.mode.time_remaining = self.match_time_remaining

      goal_event = self.sim.step(dt)

      if hasattr(self.sim, "mode"):
        self.sim.mode.state = "PLAYING"

      # Continuous in-match goal handling: award point & restart from center
      if goal_event is not None:
        scored = goal_event == f"{self.learner_team}_goal"
        total_reward += 1.0 if scored else -1.0
        self._reset_kickoff()
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