from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConfig:
    # Default Pitch Geometry
    DEFAULT_PITCH_WIDTH: float = 1200.0
    DEFAULT_PITCH_HEIGHT: float = 800.0

    WIDTH_MARGIN: float = 80.0
    HEIGHT_MARGIN: float = 40.0
    OUTER_MARGIN: float = 120.0

    # Goal & Post Geometry
    GOAL_WIDTH: float = 100.0
    GOAL_HEIGHT: float = 200.0
    GOAL_DEPTH: float = 80.0
    CENTER_CIRCLE_RADIUS: float = 100.0
    POST_RADIUS: float = 8.0

    # Global Ball Physics
    BALL_RADIUS: float = 15.0
    BALL_MASS: float = 0.35
    BALL_FRICTION: float = 0.975
    BALL_RESTITUTION: float = 0.80
    BALL_PLAYER_RESTITUTION: float = 0.20