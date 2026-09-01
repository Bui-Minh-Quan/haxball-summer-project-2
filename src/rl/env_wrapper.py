import os
import glob
import random
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.match_config import MatchConfig
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.rl.obs_extractor import extract_obs
from src.rl.ppo_core import ActorCritic
from src.rl.reward_shapers import BaseRewardShaper
from src.rl.reset_strategies import BaseResetStrategy


class ActionPlaceholder(Controller):
    def __init__(self):
        self.action = (Vec2(0, 0), False)

    def get_action(self, player_idx: int, sim) -> tuple[Vec2, bool]:
        return self.action


class RandomController(Controller):
    def __init__(self):
        self._action_to_dir = [
            Vec2(0, 0), Vec2(0, -1), Vec2(0, 1),
            Vec2(-1, 0), Vec2(1, 0), Vec2(-1, -1),
            Vec2(1, -1), Vec2(-1, 1), Vec2(1, 1),
        ]

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        move = self._action_to_dir[random.randint(0, 8)]
        kick = random.random() < 0.2
        return move, kick


class Stage2OpponentController(Controller):
  """Smooth curriculum: 90% Heuristic, 10% Random."""

  def __init__(self, team: str = "blue"):
    self.heuristic = HeuristicBotController(TeamHeuristicCoordinator(team=team))
    self.random = RandomController()
    self.is_heuristic = True

  def reset_opponent(self):
    self.is_heuristic = random.random() < 0.90

  def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
    if self.is_heuristic:
      return self.heuristic.get_action(player_idx, sim)
    return self.random.get_action(player_idx, sim)


class PoolController(Controller):
    def __init__(self, pool_dir: str, device: str = "cpu"):
        torch.set_num_threads(1)
        self.pool_dir = pool_dir
        self.device = torch.device(device)
        self.rl_model = ActorCritic(obs_dim=80).to(self.device)
        self.rl_model.eval()
        self.heuristic_ctrl = HeuristicBotController(TeamHeuristicCoordinator(team="blue"))
        self.current_mode = "heuristic"
        self.is_ready = False

        self._ego_dirs = [
            (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
            (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
            (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
        ]

    def reset_opponent(self):
        os.makedirs(self.pool_dir, exist_ok=True)
        p = random.random()

        if p < 0.30:
            self.current_mode = "heuristic"
            self.is_ready = True
            return

        self.current_mode = "rl"
        target_file = os.path.join(self.pool_dir, "latest.pt")

        if p >= 0.80:
            history_files = glob.glob(os.path.join(self.pool_dir, "history_*.pt"))
            if history_files:
                target_file = random.choice(history_files)

        if target_file and os.path.exists(target_file):
            try:
                ckpt = torch.load(target_file, map_location=self.device, weights_only=False)
                self.rl_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                self.is_ready = True
            except Exception:
                self.current_mode = "heuristic"
        else:
            self.current_mode = "heuristic"

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        if self.current_mode == "heuristic" or not self.is_ready:
            return self.heuristic_ctrl.get_action(player_idx, sim)

        player = sim.all_players[player_idx]
        obs = extract_obs(sim, player, team="blue")
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.rl_model.get_action_and_value(obs_tensor, deterministic=False)

        act = action.squeeze(0).cpu().numpy()
        m_idx = int(act[0])
        kick = bool(act[1])

        # Team Blue: sign = -1.0
        ego_x, ego_y = self._ego_dirs[m_idx]
        world_move = Vec2(ego_x * -1.0, ego_y)
        return world_move, kick


class MatchEnv(gym.Env):
    def __init__(
        self,
        match_config: MatchConfig,
        reward_shaper: BaseRewardShaper,
        reset_strategy: BaseResetStrategy,
        learner_team: str = "red",
        max_steps: int = 1800,
        frame_skip: int = 1,
    ):
        super().__init__()
        self.match_config = match_config
        self.reward_shaper = reward_shaper
        self.reset_strategy = reset_strategy
        self.learner_team = learner_team
        self.sign = 1.0 if learner_team == "red" else -1.0
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.current_step = 0
        self.episode_reward = 0.0

        self.rl_controller = ActionPlaceholder()
        for slot in self.match_config.roster:
            if slot.team == self.learner_team and slot.controller == "RL":
                slot.controller = self.rl_controller

        self.sim = Simulation(match_config=self.match_config)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(80,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 2])

        self._ego_dirs = [
            (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
            (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
            (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0

        for slot in self.match_config.roster:
            if hasattr(slot.controller, "reset_opponent"):
                slot.controller.reset_opponent()

        self.reset_strategy.reset(self.sim)
        self.reward_shaper.reset(self.sim)
        agent = self.sim.red_team[0] if self.learner_team == "red" else self.sim.blue_team[0]
        return extract_obs(self.sim, agent, self.learner_team), {}

    def step(self, action):
        self.current_step += 1

        # Transform ego action to world movement
        m_idx = int(action[0])
        kick_val = bool(action[1])
        ego_x, ego_y = self._ego_dirs[m_idx]
        world_move = Vec2(ego_x * self.sign, ego_y)

        self.rl_controller.action = (world_move, kick_val)

        accumulated_reward = 0.0
        goal_occurred = False
        goal_type = None
        truncated = self.current_step >= self.max_steps
        dt = 1.0 / 60.0

        for _ in range(self.frame_skip):
            step_goal = self.sim.step(dt)
            if step_goal is not None:
                goal_occurred = True
                goal_type = step_goal

            r, _, _ = self.reward_shaper.compute_reward(self.sim, step_goal, truncated)
            accumulated_reward += r

            if goal_occurred:
                break

        self.episode_reward += accumulated_reward
        agent = self.sim.red_team[0] if self.learner_team == "red" else self.sim.blue_team[0]
        obs = extract_obs(self.sim, agent, self.learner_team)

        if goal_occurred:
            self.reset_strategy.reset(self.sim)
            self.reward_shaper.reset(self.sim)

        info = {
            "episode_reward": float(self.episode_reward),
            "goal_event": goal_type,
        }

        agent = (
            self.sim.red_team[0]
            if self.learner_team == "red"
            else self.sim.blue_team[0]
        )

        obs = extract_obs(self.sim, agent, self.learner_team)

        return obs, accumulated_reward, False, truncated, info