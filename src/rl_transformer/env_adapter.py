from collections import deque
from collections.abc import Callable
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.engine.controllers import Controller
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_transformer.entity_obs import (
    BALL_DIM,
    EGO_DIM,
    MAX_OPPONENTS,
    MAX_TEAMMATES,
    PLAYER_DIM,
    TOTAL_TOKENS,
    extract_entity_obs,
    extract_global_critic_entities,
)

OpponentStatsType = (
    tuple[float, float]
    | list[tuple[float, float]]
    | Callable[[], tuple[float, float]]
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
  """Flexible Soccer Environment producing structured Entity-Tokens for Transformer MAPPO."""

  metadata = {"render_modes": []}

  def __init__(
      self,
      team_size: int = 3,
      learner_team_size: int | None = None,
      opp_team_size: int | None = None,
      learner_team: str = "red",
      max_round_steps: int = 3600,
      action_repeat: int = 10,
      goal_height: float | None = None,
      pitch_width: float = 1200.0,
      pitch_height: float = 800.0,
      opponent_controller: Controller | None = None,
      opponent_stats: OpponentStatsType = (3200.0, 1200.0),
      random_reset_opponents: list[str] | None = None,
      frame_stack: int = 3,
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
    self.opponent_stats = opponent_stats
    self.random_reset_opponents = set(random_reset_opponents or ["random"])

    # Temporal Frame Stacking Deque
    self.frame_stack = frame_stack
    self.obs_history = deque(maxlen=self.frame_stack)

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

    actor_shape_prefix = () if self.learner_team_size == 1 else (self.learner_team_size,)

    self.observation_space = spaces.Dict({
        "actor_ego": spaces.Box(
            -np.inf, np.inf, shape=actor_shape_prefix + (EGO_DIM,), dtype=np.float32
        ),
        "actor_ball": spaces.Box(
            -np.inf, np.inf, shape=actor_shape_prefix + (BALL_DIM,), dtype=np.float32
        ),
        "actor_teammates": spaces.Box(
            -np.inf, np.inf, shape=actor_shape_prefix + (MAX_TEAMMATES, PLAYER_DIM), dtype=np.float32
        ),
        "actor_opponents": spaces.Box(
            -np.inf, np.inf, shape=actor_shape_prefix + (MAX_OPPONENTS, PLAYER_DIM), dtype=np.float32
        ),
        "actor_mask": spaces.Box(
            0, 1, shape=actor_shape_prefix + (TOTAL_TOKENS,), dtype=bool
        ),
        "critic_ball": spaces.Box(-np.inf, np.inf, shape=(BALL_DIM,), dtype=np.float32),
        "critic_learners": spaces.Box(-np.inf, np.inf, shape=(3, PLAYER_DIM), dtype=np.float32),
        "critic_opponents": spaces.Box(-np.inf, np.inf, shape=(3, PLAYER_DIM), dtype=np.float32),
        "critic_mask": spaces.Box(0, 1, shape=(7,), dtype=bool),
    })

    if self.learner_team_size == 1:
      self.action_space = spaces.MultiDiscrete([9, 2])
    else:
      self.action_space = spaces.MultiDiscrete([[9, 2]] * self.learner_team_size)

    self.current_step = 0
    self.match_time_remaining = float(self.max_round_steps) / 60.0
    self.sim: Simulation | None = None
    self.learner_slots: list[tuple[int, PlayerSlot]] = []
    self.opp_slots: list[tuple[int, PlayerSlot]] = []

    self._init_simulation()

  def _is_random_opponent(self) -> bool:
    mode = getattr(self.opponent_controller, "current_mode", None)
    return mode == "random" or isinstance(self.opponent_controller, RandomOpponentController)

  def _sample_opponent_stats(self) -> tuple[float, float]:
    mode = getattr(self.opponent_controller, "current_mode", "heuristic")
    if mode != "heuristic":
      return 3200.0, 1200.0

    if callable(self.opponent_stats):
      return self.opponent_stats()
    if isinstance(self.opponent_stats, list) and self.opponent_stats:
      return random.choice(self.opponent_stats)
    if isinstance(self.opponent_stats, tuple):
      return self.opponent_stats
    return 3200.0, 1200.0

  def _init_simulation(self):
    roster = []
    self.learner_slots = []
    self.opp_slots = []

    for i in range(self.learner_team_size):
      ph = ActionPlaceholder()
      slot = PlayerSlot(self.learner_team, PlayerStats(f"L_{i+1}", accel=3200.0), ph)
      roster.append(slot)
      self.learner_slots.append((i, slot))

    for j in range(self.opp_team_size):
      ph_opp = ActionPlaceholder()
      slot = PlayerSlot(self.opp_team, PlayerStats(f"O_{j+1}", accel=3200.0), ph_opp)
      roster.append(slot)
      self.opp_slots.append((self.learner_team_size + j, slot))

    match_cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=self.max_round_steps / 60.0, score_limit=99),
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
      if hasattr(self.sim.mode, "score_limit"):
        self.sim.mode.score_limit = 999
      if hasattr(self.sim.mode, "reset_positions"):
        self.sim.mode.reset_positions = lambda *args, **kwargs: None


  def _should_random_reset(self) -> bool:
    """Checks if the active opponent mode matches the random reset criteria."""
    if "all" in self.random_reset_opponents:
      return True

    # Identify opponent mode
    if isinstance(self.opponent_controller, RandomOpponentController):
      curr_mode = "random"
    else:
      curr_mode = getattr(self.opponent_controller, "current_mode", "heuristic")

    return curr_mode in self.random_reset_opponents

  def _reset_kickoff(self):
    p = self.sim.pitch
    c = self.sim.center
    safe_m = 50.0
    min_dist = 52.0

    # 1. Full-pitch chaotic spawn (Triggered if opponent matches configured modes)
    if self._should_random_reset():
      self.sim.ball.pos = Vec2(
          random.uniform(p.left + safe_m, p.right - safe_m),
          random.uniform(p.top + safe_m, p.bottom - safe_m),
      )
      self.sim.ball.vel = Vec2(0.0, 0.0)

      placed_positions: list[Vec2] = [self.sim.ball.pos]
      all_players = self.sim.red_team + self.sim.blue_team

      for pl in all_players:
        placed = False
        for _ in range(50):
          cand = Vec2(
              random.uniform(p.left + safe_m, p.right - safe_m),
              random.uniform(p.top + safe_m, p.bottom - safe_m),
          )
          if all(cand.distance_to(pos) >= min_dist for pos in placed_positions):
            pl.pos = cand
            placed_positions.append(cand)
            placed = True
            break

        if not placed:
          pl.pos = Vec2(
              random.uniform(p.left + safe_m, p.right - safe_m),
              random.uniform(p.top + safe_m, p.bottom - safe_m),
          )
          placed_positions.append(pl.pos)

        pl.vel = Vec2(0.0, 0.0)
        pl.kick_cooldown_timer = 0.0
      return

    # 2. Standard Structured Kickoff (Clean symmetrical half-pitch spawns)
    self.sim.ball.pos = Vec2(c.x, c.y)
    self.sim.ball.vel = Vec2(0.0, 0.0)

    placed_positions: list[Vec2] = []

    # Red Team in Left Half (x < c.x)
    for idx, pl in enumerate(self.sim.red_team):
      placed = False
      for _ in range(50):
        cand = Vec2(
            random.uniform(p.left + safe_m, c.x - 40.0),
            random.uniform(p.top + safe_m, p.bottom - safe_m),
        )
        if all(cand.distance_to(pos) >= min_dist for pos in placed_positions):
          pl.pos = cand
          placed_positions.append(cand)
          placed = True
          break
      if not placed:
        fallback_x = c.x - 100.0 - (idx * 80.0)
        fallback_y = c.y + (70.0 if idx % 2 == 1 else -70.0)
        pl.pos = Vec2(max(p.left + safe_m, fallback_x), fallback_y)
        placed_positions.append(pl.pos)
      pl.vel = Vec2(0.0, 0.0)
      pl.kick_cooldown_timer = 0.0

    # Blue Team in Right Half (x > c.x)
    for idx, pl in enumerate(self.sim.blue_team):
      placed = False
      for _ in range(50):
        cand = Vec2(
            random.uniform(c.x + 40.0, p.right - safe_m),
            random.uniform(p.top + safe_m, p.bottom - safe_m),
        )
        if all(cand.distance_to(pos) >= min_dist for pos in placed_positions):
          pl.pos = cand
          placed_positions.append(cand)
          placed = True
          break
      if not placed:
        fallback_x = c.x + 100.0 + (idx * 80.0)
        fallback_y = c.y + (70.0 if idx % 2 == 1 else -70.0)
        pl.pos = Vec2(min(p.right - safe_m, fallback_x), fallback_y)
        placed_positions.append(pl.pos)
      pl.vel = Vec2(0.0, 0.0)
      pl.kick_cooldown_timer = 0.0

  def _extract_single_payload(self) -> dict[str, np.ndarray]:
    """Extracts instantaneous single-frame entity tokens."""
    squad = self.sim.red_team if self.learner_team == "red" else self.sim.blue_team
    actor_tokens = [extract_entity_obs(self.sim, player, self.learner_team) for player in squad]

    if self.learner_team_size == 1:
      act_data = actor_tokens[0]
      ego_arr, ball_arr = act_data["ego"], act_data["ball"]
      mates_arr, opps_arr = act_data["teammates"], act_data["opponents"]
      mask_arr = act_data["key_padding_mask"]
    else:
      ego_arr = np.stack([t["ego"] for t in actor_tokens], axis=0)
      ball_arr = np.stack([t["ball"] for t in actor_tokens], axis=0)
      mates_arr = np.stack([t["teammates"] for t in actor_tokens], axis=0)
      opps_arr = np.stack([t["opponents"] for t in actor_tokens], axis=0)
      mask_arr = np.stack([t["key_padding_mask"] for t in actor_tokens], axis=0)

    critic_data = extract_global_critic_entities(self.sim, self.learner_team)
    return {
        "actor_ego": ego_arr,
        "actor_ball": ball_arr,
        "actor_teammates": mates_arr,
        "actor_opponents": opps_arr,
        "actor_mask": mask_arr,
        "critic_ball": critic_data["ball"],
        "critic_learners": critic_data["learners"],
        "critic_opponents": critic_data["opponents"],
        "critic_mask": critic_data["key_padding_mask"],
    }

  def _get_obs_payload(self) -> dict[str, np.ndarray]:
    """Extracts current frame and concatenates history along the feature dimension."""
    curr = self._extract_single_payload()

    # Self-healing prime: guarantees history always has exactly K frames
    if len(self.obs_history) == 0:
      for _ in range(self.frame_stack):
        self.obs_history.append(curr)
    else:
      self.obs_history.append(curr)

    stacked = {
        "actor_mask": curr["actor_mask"],
        "critic_mask": curr["critic_mask"],
    }
    for key in (
        "actor_ego",
        "actor_ball",
        "actor_teammates",
        "actor_opponents",
        "critic_ball",
        "critic_learners",
        "critic_opponents",
    ):
      stacked[key] = np.concatenate([f[key] for f in self.obs_history], axis=-1)

    return stacked

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

      accel, kick = self._sample_opponent_stats()
      opp_squad = self.sim.blue_team if self.learner_team == "red" else self.sim.red_team
      for player in opp_squad:
        player.stats.accel = accel
        player.stats.kick_strength = kick

      if hasattr(self.sim.mode, "time_remaining"):
        self.sim.mode.time_remaining = self.match_time_remaining
        self.sim.mode.state = "PLAYING"

      self._reset_kickoff()
      self.obs_history.clear()
      return self._get_obs_payload(), {}

  def step(self, action):
    dt = 1.0 / 60.0
    sign = 1.0 if self.learner_team == "red" else -1.0
    actions = [action] if self.learner_team_size == 1 else action

    # 1. Update Learner Actions
    for i, (_, slot) in enumerate(self.learner_slots):
      m_idx = int(actions[i][0])
      kick = bool(actions[i][1])
      ego_x, ego_y = self._ego_dirs[m_idx]
      slot.controller.action = (Vec2(ego_x * sign, ego_y), kick)

    # 2. Update Opponent Actions
    for j, slot in self.opp_slots:
      slot.controller.action = self.opponent_controller.get_action(j, self.sim)

    total_reward = 0.0

    # 3. Physics Substeps
    for _ in range(self.action_repeat):
      self.current_step += 1
      self.match_time_remaining = max(0.0, self.match_time_remaining - dt)

      if hasattr(self.sim.mode, "time_remaining"):
        self.sim.mode.time_remaining = self.match_time_remaining

      goal_event = self.sim.step(dt)

      if hasattr(self.sim, "mode"):
        self.sim.mode.state = "PLAYING"
        if hasattr(self.sim.mode, "score_limit"):
          self.sim.mode.score_limit = 999

      if goal_event is not None:
        scored = goal_event == f"{self.learner_team}_goal"
        total_reward += 1.0 if scored else -2.0
        self._reset_kickoff()
        self.obs_history.clear()
        if hasattr(self.opponent_controller, "history"):
          self.opponent_controller.history.clear()
        break

      if (
          self.current_step >= self.max_round_steps
          or self.match_time_remaining <= 0.0
      ):
        break

    # 4. Truncation & Differential Result
    terminated = False
    truncated = (
        self.current_step >= self.max_round_steps
        or self.match_time_remaining <= 0.0
    )

    if truncated:
      my_score = self.sim.score_red if self.learner_team == "red" else self.sim.score_blue
      opp_score = self.sim.score_blue if self.learner_team == "red" else self.sim.score_red
      diff = my_score - opp_score

      if diff > 0:
        total_reward += 1.0 + 0.1 * diff
      elif diff < 0:
        total_reward -= 1.0 + 0.1 * abs(diff)
      else:
        total_reward -= 0.8

    info = {
        "score_red": self.sim.score_red,
        "score_blue": self.sim.score_blue,
        "match_time_remaining": self.match_time_remaining,
    }

    return self._get_obs_payload(), float(total_reward), terminated, truncated, info