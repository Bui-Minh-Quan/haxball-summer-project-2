from abc import ABC, abstractmethod
from typing import Any
from src.engine.vector import Vec2

from pathlib import Path
from src.rl.numpy_actor import NumpyActor
from src.rl.obs_extractor import extract_obs

class Controller(ABC):
    """Abstract interface for entity input (Human, Heuristic, RL, Network)."""

    @abstractmethod
    def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
        """Returns: (move_direction: Vec2, is_kick_pressed: bool)"""
        pass


class HeuristicBotController(Controller):
    """Wraps TeamHeuristicCoordinator into the engine controller interface."""

    def __init__(self, coordinator_instance: Any):
        self.coordinator = coordinator_instance

    def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
        player = sim.all_players[player_idx]
        return self.coordinator.get_action(player, sim)


class NumpyRLController(Controller):
    """Pure NumPy inference controller. Zero PyTorch or CUDA dependencies."""

    def __init__(self, actor: NumpyActor | str | Path, team: str):
        if isinstance(actor, (str, Path)):
            self.actor = NumpyActor(str(actor))
        else:
            self.actor = actor

        self.team = team.lower()

        # Ego-centric directions: index -> (ego_x, ego_y) matching PPO training
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

    def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
        player = sim.all_players[player_idx]
        obs = extract_obs(sim, player, self.team)

        m_idx, kick_idx = self.actor.forward(obs)

        # Map ego-centric action to world space
        sign = 1.0 if self.team == "red" else -1.0
        ego_x, ego_y = self._ego_dirs[m_idx]
        world_move = Vec2(ego_x * sign, ego_y)
        kick = bool(kick_idx == 1)

        return world_move, kick