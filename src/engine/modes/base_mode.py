from abc import ABC, abstractmethod
from typing import Any


class GameMode(ABC):
    """Defines lifecycle rules, boundary rules, and win conditions for a match."""

    @abstractmethod
    def init_mode(self, sim: Any):
        """Called once when simulation initializes."""
        pass

    @abstractmethod
    def on_step(self, sim: Any, dt: float) -> str | None:
        """Called on every simulation step to advance state timers and score checks."""
        pass

    @abstractmethod
    def enforce_player_bounds(self, player: Any, sim: Any):
        """Enforces custom boundary constraints on a player."""
        pass

    @abstractmethod
    def enforce_ball_bounds(self, sim: Any) -> str | None:
        """Enforces custom boundary constraints and goal triggers on the ball."""
        pass

    @abstractmethod
    def is_game_over(self, sim: Any) -> bool:
        """Evaluates time limit and score limit conditions."""
        pass