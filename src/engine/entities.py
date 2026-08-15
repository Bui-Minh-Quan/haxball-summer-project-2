import math
from config.physics_config import PhysicsConfig
from src.engine.vector import Vec2


class Disc:
    """Base circular physics body."""
    def __init__(self, pos: Vec2, radius: float, mass: float, friction: float, restitution: float):
        self.pos = pos.copy()
        self.vel = Vec2(0.0, 0.0)
        self.radius = radius
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.friction = friction
        self.restitution = restitution

    def step(self, dt: float):
        self.pos += self.vel * dt
        # Frame-rate and sub-step independent friction calculation
        self.vel *= math.pow(self.friction, dt * 60.0)


class Player(Disc):
    def __init__(self, pos: Vec2, team: str, cfg: PhysicsConfig):
        super().__init__(
            pos=pos,
            radius=cfg.PLAYER_RADIUS,
            mass=cfg.PLAYER_MASS,
            friction=cfg.PLAYER_FRICTION,
            restitution=cfg.PLAYER_RESTITUTION,
        )
        self.team = team 
        self.cfg = cfg
        self.accel = Vec2(0.0, 0.0)
        self.is_kicking = False
        self.kick_timer = 0.0

    def apply_input(self, move_dir: Vec2, kick_pressed: bool, dt: float):
        if move_dir.length_sq() > 0:
            # Apply PLAYER_SPEED_MULT
            effective_accel = self.cfg.PLAYER_ACCEL * self.cfg.PLAYER_SPEED_MULT
            self.accel = move_dir.normalize() * effective_accel
        else:
            self.accel = Vec2(0.0, 0.0)

        self.vel += self.accel * dt

        self.kick_timer = max(0.0, self.kick_timer - dt)
        self.is_kicking = kick_pressed and (self.kick_timer == 0.0)
        if self.is_kicking:
            self.kick_timer = self.cfg.KICK_COOLDOWN


class Ball(Disc):
    def __init__(self, pos: Vec2, cfg: PhysicsConfig):
        super().__init__(
            pos=pos,
            radius=cfg.BALL_RADIUS,
            mass=cfg.BALL_MASS,
            friction=cfg.BALL_FRICTION,
            restitution=cfg.BALL_RESTITUTION,
        )


class GoalPost(Disc):
    def __init__(self, pos: Vec2, cfg: PhysicsConfig):
        super().__init__(
            pos=pos,
            radius=cfg.POST_RADIUS,
            mass=0.0, 
            friction=1.0,
            restitution=cfg.BALL_RESTITUTION,
        )