import math
import random
from abc import ABC, abstractmethod
from src.engine.simulation import Simulation
from src.engine.vector import Vec2

class BaseResetStrategy(ABC):
    @abstractmethod
    def reset(self, sim: Simulation):
        pass



class RandomReset(BaseResetStrategy):
    """Dynamically positions multi-agent rosters (1v1, 2v2, 3v3).

    Assigns role-based spawn radii (Striker near ball, Defender backline)
    to guarantee proper spacing on kickoff.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed) if seed is not None else random

    def set_seed(self, seed: int):
        self.rng = random.Random(seed)

    def reset(self, sim: Simulation):
        p = sim.pitch
        ball = sim.ball
        sim.reset_positions()

        if hasattr(sim.mode, "state"):
            sim.mode.state = "PLAYING"
            sim.mode.kickoff_timer = 0.0

        # Ball spawned in midfield zone
        ball.pos.x = self.rng.uniform(sim.center.x - 100.0, sim.center.x + 100.0)
        ball.pos.y = self.rng.uniform(p.top + 100.0, p.bottom - 100.0)
        ball.vel = Vec2(0.0, 0.0)

        # 1. Reset Red Squad (West side of the ball)
        for idx, red in enumerate(sim.red_team):
            angle_r = self.rng.uniform(0.65 * math.pi, 1.35 * math.pi)
            # Player 0 presses forward; teammates form depth/support lines
            dist_r = self.rng.uniform(80.0, 150.0) if idx == 0 else self.rng.uniform(170.0, 260.0)
            
            red.pos = ball.pos + Vec2(math.cos(angle_r) * dist_r, math.sin(angle_r) * dist_r)
            red.pos.x = max(p.left + red.radius + 10.0, min(p.right - red.radius - 10.0, red.pos.x))
            red.pos.y = max(p.top + red.radius + 10.0, min(p.bottom - red.radius - 10.0, red.pos.y))
            red.vel = Vec2(0.0, 0.0)
            red.kick_cooldown_timer = 0.0

        # 2. Reset Blue Squad (East side of the ball)
        for idx, blue in enumerate(sim.blue_team):
            angle_b = self.rng.uniform(-0.35 * math.pi, 0.35 * math.pi)
            dist_b = self.rng.uniform(80.0, 150.0) if idx == 0 else self.rng.uniform(170.0, 260.0)

            blue.pos = ball.pos + Vec2(math.cos(angle_b) * dist_b, math.sin(angle_b) * dist_b)
            blue.pos.x = max(p.left + blue.radius + 10.0, min(p.right - blue.radius - 10.0, blue.pos.x))
            blue.pos.y = max(p.top + blue.radius + 10.0, min(p.bottom - blue.radius - 10.0, blue.pos.y))
            blue.vel = Vec2(0.0, 0.0)
            blue.kick_cooldown_timer = 0.0