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
    def __init__(self, pool_dir: str, team: str = "blue", device: str = "cpu"):
        torch.set_num_threads(1)
        self.pool_dir = pool_dir
        self.team = team
        self.sign = 1.0 if team == "red" else -1.0
        self.device = torch.device(device)
        self.rl_model = ActorCritic(obs_dim=80).to(self.device)
        self.rl_model.eval()
        self.heuristic_ctrl = HeuristicBotController(
            TeamHeuristicCoordinator(team=team)
        )
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
        reward_shaper,
        reset_strategy,
        learner_team: str = "red",
        max_steps: int = 1800,
    ):
        self.match_config = match_config
        self.sim = Simulation(match_config)
        self.reward_shaper = reward_shaper
        self.reset_strategy = reset_strategy
        self.learner_team = learner_team
        self.max_steps = max_steps
        self.current_step = 0

        self._EGO_DIRS = [
            (0.0, 0.0), (0.0, -1.0), (0.0, 1.0),
            (-1.0, 0.0), (1.0, 0.0), (-1.0, -1.0),
            (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
        ]

        # Identify active learner slots and inject ActionPlaceholder ONCE
        self.learner_slots = []
        for i, slot in enumerate(match_config.roster):
            if slot.team == learner_team and slot.controller == "RL":
                slot.controller = ActionPlaceholder()
                self.learner_slots.append((i, slot))
                
        self.num_learners = len(self.learner_slots)
        
        # Adjust Gym spaces based on team size
        obs_dim = 80
        if self.num_learners == 1:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([9, 2])
        else:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_learners, obs_dim), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([[9, 2]] * self.num_learners)


    def _get_obs(self):
        obs_list = []
        for roster_idx, slot in self.learner_slots:
            # Player index inside their specific team array
            team_roster = [s for s in self.match_config.roster if s.team == self.learner_team]
            player_idx_in_team = team_roster.index(slot)
            agent = self.sim.red_team[player_idx_in_team] if self.learner_team == "red" else self.sim.blue_team[player_idx_in_team]
            obs_list.append(extract_obs(self.sim, agent, self.learner_team))
            
        if self.num_learners == 1:
            return obs_list[0]
        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        for slot in self.match_config.roster:
            if hasattr(slot.controller, "reset_opponent"):
                slot.controller.reset_opponent()

        self.reset_strategy.reset(self.sim)
        self.reward_shaper.reset(self.sim)
        
        # Use the multi-agent observation extractor
        return self._get_obs(), {}

    def step(self, action):
        # Route actions to the correct learner slots
        if self.num_learners == 1:
            action = [action]

        for i, (roster_idx, slot) in enumerate(self.learner_slots):
            m_idx = int(action[i][0])
            k_val = bool(action[i][1])
            ego_x, ego_y = self._EGO_DIRS[m_idx]
            sign = 1.0 if self.learner_team == "red" else -1.0
            world_move = Vec2(ego_x * sign, ego_y)
            
            # Update the pre-existing placeholder
            slot.controller.action = (world_move, k_val)

        goal_event = self.sim.step(1.0 / 60.0)
        self.current_step += 1

        obs = self._get_obs()
        truncated = self.current_step >= self.max_steps
        
        # Shared Team Reward
        reward, terminated, info = self.reward_shaper.compute_reward(
            self.sim, goal_event, truncated
        )

        # Distribute shared reward to all agents
        if self.num_learners > 1:
            reward = np.full(self.num_learners, reward, dtype=np.float32)

        if goal_event is not None or truncated:
            self.reset_strategy.reset(self.sim)
            self.reward_shaper.reset(self.sim)

        return obs, reward, terminated, truncated, info