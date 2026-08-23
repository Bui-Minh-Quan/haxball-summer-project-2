import os
import glob
import random
import torch
from src.rl.ppo_core import ActorCritic
from src.rl.obs_extractor import extract_universal_obs
from src.engine.controllers import Controller, HeuristicBotController
from src.bots.heuristic_bot import TeamHeuristicCoordinator



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


# Support function
def extract_role_obs(sim: Simulation, player: Any, team: str, role: str) -> np.ndarray:
    base_obs = extract_universal_obs(sim, player, team)
    role_map = {
        "ST": [1.0, 0.0, 0.0, 0.0],
        "CM": [0.0, 1.0, 0.0, 0.0],
        "CB": [0.0, 0.0, 1.0, 0.0],
        "GK": [0.0, 0.0, 0.0, 1.0],
    }
    role_vec = np.array(role_map.get(role.upper(), [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    return np.concatenate([base_obs, role_vec])

# Controllers
class DummyRLController(Controller):
    def __init__(self):
        self.action = (Vec2(0, 0), False)

    def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
        return self.action

class MultiAgentDummyController(Controller):
    def __init__(self):
        self.action = (Vec2(0, 0), False)

    def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
        return self.action

class LeagueOpponentController(Controller):
    def __init__(self, pool_dir: str, device: str = "cpu", obs_dim: int = 84):
        # Force single-threaded execution per worker to prevent CPU locking
        torch.set_num_threads(1)
        self.pool_dir = pool_dir
        self.device = torch.device(device)
        self.rl_model = ActorCritic(obs_dim=obs_dim).to(self.device)
        self.rl_model.eval()
        self.heuristic_ctrl = HeuristicBotController(TeamHeuristicCoordinator(team="blue"))
        self.current_mode = "heuristic"
        self.is_ready = False
        self.loaded_file = None

        self._action_to_dir = {
            0: Vec2(0, 0), 1: Vec2(0, -1), 2: Vec2(0, 1),
            3: Vec2(-1, 0), 4: Vec2(1, 0), 5: Vec2(-1, -1),
            6: Vec2(1, -1), 7: Vec2(-1, 1), 8: Vec2(1, 1),
        }

    def reset_opponent(self):
        p = random.random()
        if p < 0.45:
            self.current_mode = "heuristic"
            self.is_ready = True
            return

        self.current_mode = "rl"
        target_file = os.path.join(self.pool_dir, "latest.pt")
        
        if p >= 0.80:
            history_files = glob.glob(os.path.join(self.pool_dir, "history_*.pt"))
            if history_files:
                target_file = random.choice(history_files)

        # Only reload from disk if target weights changed
        if target_file and os.path.exists(target_file):
            try:
                ckpt = torch.load(target_file, map_location=self.device, weights_only=False)
                self.rl_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                self.is_ready = True
                self.loaded_file = target_file
            except Exception:
                self.current_mode = "heuristic"
        else:
            self.current_mode = "heuristic"

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        if self.current_mode == "heuristic" or not self.is_ready:
            return self.heuristic_ctrl.get_action(player_idx, sim)

        player = sim.all_players[player_idx]
        role = getattr(sim.match_config.roster[player_idx], "role", "ST")
        obs = extract_role_obs(sim, player, team="blue", role=role)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.rl_model.get_action_and_value(obs_tensor, deterministic=True)

        act = action.squeeze(0).cpu().numpy()
        move_dir = self._action_to_dir.get(int(act[0]), Vec2(0, 0))
        is_kick = bool(act[1])

        return move_dir, is_kick


# Environments
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

        for global_idx, slot in enumerate(self.match_config.roster):
            if slot.controller == "RL":
                ctrl = MultiAgentDummyController()
                slot.controller = ctrl
                self.rl_slots.append({
                    "team": slot.team,
                    "global_idx": global_idx,
                    "role": getattr(slot, "role", "ST"),
                })
                self.rl_controllers.append(ctrl)

        self.num_agents = len(self.rl_slots)
        self.sim = Simulation(match_config=self.match_config)

        for slot in self.rl_slots:
            slot["player"] = self.sim.all_players[slot["global_idx"]]

        if self.num_agents == 1:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(84,), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([9, 2])
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, 84), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([9, 2] * self.num_agents)

        self._action_to_dir = {
            0: Vec2(0, 0), 1: Vec2(0, -1), 2: Vec2(0, 1),
            3: Vec2(-1, 0), 4: Vec2(1, 0), 5: Vec2(-1, -1),
            6: Vec2(1, -1), 7: Vec2(-1, 1), 8: Vec2(1, 1),
        }

    def _get_obs(self) -> np.ndarray:
        obs_list = [
            extract_role_obs(self.sim, slot["player"], slot["team"], slot["role"])
            for slot in self.rl_slots
        ]
        if self.num_agents == 1:
            return obs_list[0]
        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reset_strategy.reset(self.sim)
        self.reward_shaper.reset(self.sim)
        return self._get_obs(), {}

    def step(self, actions: Any):
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

        scalar_reward = float(agent_rewards[0]) if len(agent_rewards) > 0 else 0.0
        info["agent_rewards"] = np.array(agent_rewards, dtype=np.float32)
        info["goal_event"] = goal_event

        return self._get_obs(), scalar_reward, terminated, truncated, info
  
class LeagueHaxballEnv(MultiAgentHaxballEnv):
    def reset(self, seed=None, options=None):
        for slot in self.match_config.roster:
            if isinstance(slot.controller, LeagueOpponentController):
                slot.controller.reset_opponent()
        return super().reset(seed=seed, options=options)

    