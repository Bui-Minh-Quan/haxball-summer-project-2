import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.match_config import MatchConfig
from src.engine.controllers import Controller
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.obs_extractor import extract_universal_obs
from src.rl.reward_shapers import BaseRewardShaper
from src.rl.reset_strategies import BaseResetStrategy


class DummyRLController(Controller):
    def __init__(self):
        self.action = (Vec2(0, 0), False)

    def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
        return self.action


class HaxballGymEnv(gym.Env):
    """
    Standard Gymnasium Wrapper.
    Contains ZERO stage-specific conditionals or hardcoded coordinates.
    """

    def __init__(
        self,
        match_config: MatchConfig,
        reward_shaper: BaseRewardShaper,
        reset_strategy: BaseResetStrategy,
        max_steps: int = 360,  # 6 seconds at 60 FPS
    ):
        super().__init__()
        self.match_config = match_config
        self.reward_shaper = reward_shaper
        self.reset_strategy = reset_strategy
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_reward = 0.0

        # Inject controller into Red Player slot
        self.rl_controller = DummyRLController()
        self.match_config.roster[0].controller = self.rl_controller

        self.sim = Simulation(match_config=self.match_config)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(80,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 2])

        self._action_to_dir = {
            0: Vec2(0, 0), 1: Vec2(0, -1), 2: Vec2(0, 1),
            3: Vec2(-1, 0), 4: Vec2(1, 0), 5: Vec2(-1, -1),
            6: Vec2(1, -1), 7: Vec2(-1, 1), 8: Vec2(1, 1),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0

        # Delegate reset mechanics to the injected strategy
        self.reset_strategy.reset(self.sim)
        self.reward_shaper.reset(self.sim)

        agent = self.sim.red_team[0]
        return extract_universal_obs(self.sim, agent, "red"), {}

    def step(self, action):
        self.current_step += 1
        move_dir = self._action_to_dir[int(action[0])]
        is_kick = bool(action[1])
        self.rl_controller.action = (move_dir, is_kick)

        goal_event = self.sim.step(dt=1.0 / 60.0)
        truncated = self.current_step >= self.max_steps

        reward, terminated, info = self.reward_shaper.compute_reward(
            self.sim, goal_event, truncated
        )
        self.episode_reward += reward

        info["episode_reward"] = self.episode_reward
        info["steps"] = self.current_step

        agent = self.sim.red_team[0]
        obs = extract_universal_obs(self.sim, agent, "red")

        return obs, reward, terminated, truncated, info