import math
from config.match_config import PlayerStats
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
        self.vel *= math.pow(self.friction, dt * 60.0)


class Player(Disc):

    def __init__(self, pos: Vec2, team: str, stats: PlayerStats):
        super().__init__(
            pos=pos,
            radius=stats.radius,
            mass=stats.mass,
            friction=stats.friction,
            restitution=stats.restitution,
        )
        self.team = team
        self.stats = stats
        self.accel = Vec2(0.0, 0.0)
        self.is_kicking = False
        self.kick_cooldown_timer = 0.0
        self.kick_visual_timer = 0.0

    def process_input(self, move_dir: Vec2, kick_pressed: bool, dt: float):
        if move_dir.length_sq() > 0:
            self.accel = move_dir.normalize() * self.stats.accel
        else:
            self.accel = Vec2(0.0, 0.0)

        self.kick_cooldown_timer = max(0.0, self.kick_cooldown_timer - dt)
        self.kick_visual_timer = max(0.0, self.kick_visual_timer - dt)

        self.is_kicking = False
        if kick_pressed and self.kick_cooldown_timer <= 0.0:
            self.is_kicking = True
            self.kick_cooldown_timer = self.stats.kick_cooldown
            self.kick_visual_timer = 0.05

    def apply_accel(self, dt: float):
        self.vel += self.accel * dt


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