import os
import glob
import random
import torch
from src.rl.ppo_core import ActorCritic
from src.rl.obs_extractor import extract_universal_obs
from src.engine.controllers import Controller, HeuristicBotController
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.rl.env_wrapper import HaxballGymEnv


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

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

class MultiAgentDummyController(Controller):

    def __init__(self):
        self.action = (Vec2(0, 0), False)

    def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
        return self.action

class LeagueOpponentController(Controller):
    """Dynamically swaps the Blue opponent between Heuristic, Historical, and Latest RL."""
    
    def __init__(self, pool_dir: str, device="cpu"):
        self.pool_dir = pool_dir
        self.device = device
        self.rl_model = ActorCritic(obs_dim=80).to(device)
        self.rl_model.eval()
        self.heuristic_ctrl = HeuristicBotController(TeamHeuristicCoordinator(team="blue"))
        self.current_mode = "heuristic"
        self.is_ready = False

    def reset_opponent(self):
        """Called at the start of every episode to select a new opponent."""
        os.makedirs(self.pool_dir, exist_ok=True)
        
        p = random.random()
        if p < 0.20:
            # 20% Heuristic Anchor
            self.current_mode = "heuristic"
            self.is_ready = True
        else:
            self.current_mode = "rl"
            target_file = None
            
            if p < 0.70:
                # 50% Latest Self-Play Mirror
                target_file = os.path.join(self.pool_dir, "latest.pt")
            else:
                # 30% Historical Checkpoints
                history_files = glob.glob(os.path.join(self.pool_dir, "history_*.pt"))
                if history_files:
                    target_file = random.choice(history_files)
                else:
                    target_file = os.path.join(self.pool_dir, "latest.pt")
            
            # Load weights if file exists
            if target_file and os.path.exists(target_file):
                try:
                    self.rl_model.load_state_dict(torch.load(target_file, map_location=self.device, weights_only=False))
                    self.is_ready = True
                except Exception:
                    self.current_mode = "heuristic" # Fallback if read fails
            else:
                self.current_mode = "heuristic" # Fallback if pool is empty


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


class MultiAgentHaxballEnv(gym.Env):
    """Scalable Multi-Agent Environment.

    Compatible with standard Gymnasium VectorEnv runners.
    """

    def __init__(
        self,
        match_config: MatchConfig,
        reward_shaper: Any,
        reset_strategy: BaseResetStrategy,
        max_steps: int = 600,
    ):
        super().__init__()
        self.match_config = match_config
        self.reward_shaper = reward_shaper
        self.reset_strategy = reset_strategy
        self.max_steps = max_steps
        self.current_step = 0

        self.rl_slots = []
        self.rl_controllers = []

        global_idx = 0
        for slot in self.match_config.roster:
            if slot.controller == "RL":
                ctrl = MultiAgentDummyController()
                slot.controller = ctrl
                self.rl_slots.append(
                    {"team": slot.team, "global_idx": global_idx}
                )
                self.rl_controllers.append(ctrl)
            global_idx += 1

        self.num_agents = len(self.rl_slots)
        self.sim = Simulation(match_config=self.match_config)

        for slot in self.rl_slots:
            slot["player"] = self.sim.all_players[slot["global_idx"]]

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents, 80), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 2] * self.num_agents)

        self._action_to_dir = {
            0: Vec2(0, 0),
            1: Vec2(0, -1),
            2: Vec2(0, 1),
            3: Vec2(-1, 0),
            4: Vec2(1, 0),
            5: Vec2(-1, -1),
            6: Vec2(1, -1),
            7: Vec2(-1, 1),
            8: Vec2(1, 1),
        }

    def _get_obs(self):
        obs_list = []
        for slot in self.rl_slots:
            obs = extract_universal_obs(self.sim, slot["player"], slot["team"])
            obs_list.append(obs)
        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reset_strategy.reset(self.sim)
        self.reward_shaper.reset(self.sim)
        return self._get_obs(), {}

    def step(self, actions):
        self.current_step += 1
        actions_arr = np.array(actions).reshape(self.num_agents, 2)

        for i, ctrl in enumerate(self.rl_controllers):
            move_dir = self._action_to_dir[int(actions_arr[i][0])]
            is_kick = bool(actions_arr[i][1])
            ctrl.action = (move_dir, is_kick)

        goal_event = self.sim.step(dt=1.0 / 60.0)
        truncated = self.current_step >= self.max_steps

        agent_rewards, terminated, info = self.reward_shaper.compute_reward(
            self.sim, goal_event, truncated, self.rl_slots
        )

        # Return scalar float for Gym VectorEnv compatibility
        scalar_reward = (
            float(agent_rewards[0]) if len(agent_rewards) > 0 else 0.0
        )
        info["agent_rewards"] = np.array(agent_rewards, dtype=np.float32)
        info["goal_event"] = goal_event

        return self._get_obs(), scalar_reward, terminated, truncated, info


class LeagueHaxballEnv(HaxballGymEnv):
    """Triggers the opponent switch at the start of every episode."""
    def reset(self, seed=None, options=None):
        for slot in self.match_config.roster:
            if isinstance(slot.controller, LeagueOpponentController):
                slot.controller.reset_opponent()
        return super().reset(seed, options)