from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConfig:
    # --- SPEED MODIFIERS ---
    GAME_SPEED: float = 0.4        
    PLAYER_SPEED_MULT: float = 1.0 
    BALL_SPEED_MULT: float = 1.0   

    # Pitch Geometry
    PITCH_WIDTH: float = 1200.0
    PITCH_HEIGHT: float = 600.0

    WIDTH_MARGIN: float = 80.0
    HEIGHT_MARGIN: float = 40.0
    OUTER_MARGIN: float = 120.0

    GOAL_WIDTH: float = 60.0
    GOAL_HEIGHT: float = 160.0
    GOAL_DEPTH: float = 60.0        # How deep the physical net goes
    CENTER_CIRCLE_RADIUS: float = 75.0
    POST_RADIUS: float = 8.0

    # Player Physics
    PLAYER_RADIUS: float = 25.0
    PLAYER_MASS: float = 2.0
    PLAYER_ACCEL: float = 3200.0  
    PLAYER_FRICTION: float = 0.94  
    PLAYER_RESTITUTION: float = 0.75  

    # Ball Physics
    BALL_RADIUS: float = 15.0
    BALL_MASS: float = 0.35  
    BALL_FRICTION: float = 0.975  
    BALL_RESTITUTION: float = 0.80  
    BALL_PLAYER_RESTITUTION: float = 0.1  

    # Kick Mechanics
    KICK_MARGIN: float = 1.0  
    KICK_STRENGTH: float = 1000.0  
    KICK_COOLDOWN: float = 0.0