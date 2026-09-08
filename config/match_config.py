from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerStats:
    """Individual player attributes for customization, RPG progression, or difficulty scaling."""

    name: str = "Player"
    radius: float = 20.0
    mass: float = 2.0
    accel: float = 2000.0
    friction: float = 0.94
    restitution: float = 0.75
    kick_strength: float = 1200.0
    kick_cooldown: float = 0.10
    kick_margin: float = 6.0


@dataclass
class PlayerSlot:
    """Represents a single player slot in a team roster."""
    team: str
    stats: PlayerStats = field(default_factory=PlayerStats)
    controller: Any = None
    role: str = "AUTO"


@dataclass
class MatchConfig:
    """Unified configuration object to initialize any game mode or simulation."""

    mode: Any = None
    roster: list[PlayerSlot] = field(default_factory=list)
    pitch_width: float = 1200.0
    pitch_height: float = 800.0
    time_limit: float = 180.0
    kickoff_timeout: float = 10.0
    score_limit: int = 3
    game_speed: float = 0.4
    goal_height: float | None = None  # None defaults to PhysicsConfig.GOAL_HEIGHT (200.0)