from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerStats:
    """Individual player attributes for customization, RPG progression, or difficulty scaling."""

    name: str = "Player"
    radius: float = 25.0
    mass: float = 2.0
    accel: float = 3200.0
    friction: float = 0.94
    restitution: float = 0.75
    kick_strength: float = 1000.0
    kick_cooldown: float = 0.10
    kick_margin: float = 6.0


@dataclass
class PlayerSlot:
    """Represents a single player slot in a team roster."""

    team: str  # "red", "blue", "neutral"
    stats: PlayerStats = field(default_factory=PlayerStats)
    controller: Any = None  # Instance of Controller


@dataclass
class MatchConfig:
    """Unified configuration object to initialize any game mode or simulation."""

    mode: Any = None  # Instance of GameMode
    roster: list[PlayerSlot] = field(default_factory=list)
    pitch_width: float = 1200.0
    pitch_height: float = 800.0
    time_limit: float = 180.0  # seconds (0 = infinite)
    score_limit: int = 3
    game_speed: float = 1.0  # 1.0 = Normal, 0.5 = Slow-mo, 1.5 = Fast