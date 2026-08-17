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
    """Wraps HeuristicBot into the engine controller interface."""

    def __init__(self, bot_instance: Any): # type: ignore
        self.bot = bot_instance

    def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]: # pyright: ignore[reportUndefinedVariable]
        player = sim.all_players[player_idx]
        return self.bot.get_action(player, sim)