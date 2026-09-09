import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from config.physics_config import PhysicsConfig
from src.engine.controllers import Controller
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_2.obs import extract_actor_obs, extract_global_state, ACTOR_OBS_DIM, CRITIC_STATE_DIM


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
            Vec2(0, 0), Vec2(0, -1), Vec2(0, 1),
            Vec2(-1, 0), Vec2(1, 0), Vec2(-1, -1),
            Vec2(1, -1), Vec2(-1, 1), Vec2(1, 1),
        ]

    def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
        return random.choice(self._dirs), random.random() < 0.20


class MatchEnv(gym.Env):
    """Scenario-Randomized Short Rounds with MAPPO support."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        team_size: int = 1,
        learner_team: str = "red",
        max_round_steps: int = 900,  # 15s default for Stage 1 (at 60Hz)
        goal_height: float | None = None,  # None = standard 200px; 500px for wide goals
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

        # Space definitions
        if self.team_size == 1:
            self.observation_space = spaces.Dict({
                "obs": spaces.Box(-1.0, 1.0, shape=(ACTOR_OBS_DIM,), dtype=np.float32),
                "state": spaces.Box(-1.0, 1.0, shape=(CRITIC_STATE_DIM,), dtype=np.float32),
            })
            self.action_space = spaces.MultiDiscrete([9, 2])
        else:
            self.observation_space = spaces.Dict({
                "obs": spaces.Box(-1.0, 1.0, shape=(self.team_size, ACTOR_OBS_DIM), dtype=np.float32),
                "state": spaces.Box(-1.0, 1.0, shape=(CRITIC_STATE_DIM,), dtype=np.float32),
            })
            self.action_space = spaces.MultiDiscrete([[9, 2]] * self.team_size)

        self.current_step = 0
        self.match_time_remaining = 60.0
        self.sim: Simulation | None = None
        self.learner_slots: list[tuple[int, PlayerSlot]] = []
        self._init_simulation()

    def _init_simulation(self):
      roster = []
      self.learner_slots = []
      self.opp_slots = []

      # Learner slots
      for i in range(self.team_size):
        ph = ActionPlaceholder()
        slot = PlayerSlot(
            self.learner_team,
            PlayerStats(f"Learner_{i+1}", accel=3200.0),
            ph,
        )
        roster.append(slot)
        self.learner_slots.append((i, slot))

      # Opponent slots (Now also using ActionPlaceholder!)
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
          mode=ClassicMatchMode(time_limit=60.0, score_limit=99),
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

    def _sample_positions(self):
        """Randomizes player and ball positions anywhere across the entire pitch."""
        p = self.sim.pitch
        ball = self.sim.ball

        # Safe padding from the boundaries so objects don't spawn clipped inside walls
        margin_x = 60.0
        margin_y = 50.0

        # 1. Ball spawns anywhere across the pitch
        ball.pos.x = random.uniform(p.left + margin_x, p.right - margin_x)
        ball.pos.y = random.uniform(p.top + margin_y, p.bottom - margin_y)
        ball.vel = Vec2(0.0, 0.0)

        # 2. Helper to find valid spawn location across the whole arena
        def get_valid_pos(existing: list[Vec2]) -> Vec2:
            for _ in range(100):
                x = random.uniform(p.left + margin_x, p.right - margin_x)
                y = random.uniform(p.top + margin_y, p.bottom - margin_y)
                pos = Vec2(x, y)

                # Keep clear from the ball so players don't immediately trigger a collision on frame 0
                if pos.distance_to(ball.pos) < 55.0:
                    continue

                # Keep clear from other spawned players
                if any(pos.distance_to(other) < 50.0 for other in existing):
                    continue

                return pos

            # Safe fallback if rejection sampling exceeds attempts
            return Vec2(self.sim.center.x, self.sim.center.y)

        # 3. Spawn all players anywhere on the field regardless of team
        spawned: list[Vec2] = []
        for player in self.sim.all_players:
            pos = get_valid_pos(spawned)
            player.pos = pos
            player.vel = Vec2(0.0, 0.0)
            player.kick_cooldown_timer = 0.0
            spawned.append(pos)

    def _get_obs_payload(self) -> dict[str, np.ndarray]:
        team_squad = self.sim.red_team if self.learner_team == "red" else self.sim.blue_team
        obs_list = [extract_actor_obs(self.sim, player, self.learner_team) for player in team_squad]
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

        # Cycle the opponent type!
        if hasattr(self.opponent_controller, "reset_opponent"):
            self.opponent_controller.reset_opponent()

        # Stage 1: Clean 0-0 context. (Reserve random scoreboards for Stage 2+)
        if self.goal_height and self.goal_height >= 400.0:
            self.match_time_remaining = 15.0
            self.sim.score_red = 0
            self.sim.score_blue = 0
        else:
            self.match_time_remaining = random.uniform(10.0, 60.0)
            self.sim.score_red = random.randint(0, 3)
            self.sim.score_blue = random.randint(0, 3)

        if hasattr(self.sim.mode, "time_remaining"):
            self.sim.mode.time_remaining = self.match_time_remaining
            self.sim.mode.state = "PLAYING"

        self._sample_positions()
        return self._get_obs_payload(), {}

    def step(self, action):
        dt = 1.0 / 60.0
        action_repeat = 4

        # 1. Update Learner Actions
        sign = 1.0 if self.learner_team == "red" else -1.0
        actions = [action] if self.team_size == 1 else action
        for i, (_, slot) in enumerate(self.learner_slots):
            m_idx = int(actions[i][0])
            kick = bool(actions[i][1])
            ego_x, ego_y = self._ego_dirs[m_idx]
            slot.controller.action = (Vec2(ego_x * sign, ego_y), kick)

        # 2. Update Opponent Actions ONCE per decision (15 Hz)
        opp_sign = -1.0 if self.learner_team == "red" else 1.0
        for j, slot in self.opp_slots:
            opp_action = self.opponent_controller.get_action(j, self.sim)
            slot.controller.action = opp_action

        total_reward = 0.0
        goal_event = None
        terminated = False
        truncated = False

        # 3. Advance Physics (Both sides maintain actions across the 4 substeps)
        for _ in range(action_repeat):
            self.current_step += 1
            self.match_time_remaining = max(0.0, self.match_time_remaining - dt)

            if hasattr(self.sim.mode, "time_remaining"):
                self.sim.mode.time_remaining = self.match_time_remaining

            step_goal = self.sim.step(dt)
            if step_goal is not None and goal_event is None:
                goal_event = step_goal

            total_reward -= 0.0002

            is_goal = goal_event is not None
            is_round_timeout = self.current_step >= self.max_round_steps
            is_match_timeout = self.match_time_remaining <= 0.0

            terminated = is_goal
            truncated = (is_round_timeout or is_match_timeout) and not terminated

            if terminated or truncated:
                break

        # 3. Apply Goal Rewards
        if terminated:
            scored = goal_event == f"{self.learner_team}_goal"
            if scored:
                total_reward += 1.0
                if self.learner_team == "red":
                    self.sim.score_red += 1
                else:
                    self.sim.score_blue += 1
            else:
                total_reward -= 1.0
                if self.learner_team == "red":
                    self.sim.score_blue += 1
                else:
                    self.sim.score_red += 1

        # 4. Final Whistle Match Outcome Reward
        if truncated and is_match_timeout and not (self.goal_height and self.goal_height >= 400.0):
            my_score = self.sim.score_red if self.learner_team == "red" else self.sim.score_blue
            opp_score = self.sim.score_blue if self.learner_team == "red" else self.sim.score_red
            score_diff = my_score - opp_score

            if score_diff > 0:
                total_reward += 0.5 * score_diff
            elif score_diff < 0:
                total_reward -= 0.5 * abs(score_diff)
            else:
                total_reward -= 0.25

        info = {
            "goal_event": goal_event,
            "match_time_remaining": self.match_time_remaining,
            "score_red": self.sim.score_red,
            "score_blue": self.sim.score_blue,
        }

        return self._get_obs_payload(), float(total_reward), terminated, truncated, info