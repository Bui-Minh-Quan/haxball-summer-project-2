from abc import ABC, abstractmethod
from typing import Any
from src.engine.vector import Vec2


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