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
        self.kick_cooldown_timer = 0.0
        self.kick_visual_timer = 0.0  # Keeps the color change visible for ~0.12s

    def process_input(self, move_dir: Vec2, kick_pressed: bool, dt: float):
        """Called ONCE per frame before physics substeps."""
        if move_dir.length_sq() > 0:
            effective_accel = self.cfg.PLAYER_ACCEL * self.cfg.PLAYER_SPEED_MULT
            self.accel = move_dir.normalize() * effective_accel
        else:
            self.accel = Vec2(0.0, 0.0)

        self.kick_cooldown_timer = max(0.0, self.kick_cooldown_timer - dt)
        self.kick_visual_timer = max(0.0, self.kick_visual_timer - dt)

        # 1. Physics flag: Reset EVERY frame so phantom kicks don't happen
        self.is_kicking = False
        
        # 2. Trigger kick ONLY if pressed right now and cooldown is 0
        if kick_pressed and self.kick_cooldown_timer == 0.0:
            self.is_kicking = True
            self.kick_cooldown_timer = self.cfg.KICK_COOLDOWN
            self.kick_visual_timer = 0.01  # Visual feedback duration stays alive for rendering

    def apply_accel(self, dt: float):
        """Called inside micro-steps."""
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