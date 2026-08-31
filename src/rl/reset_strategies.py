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
    """Spawns ball centrally with randomized player arcs and distances.

    Accepts seed setting for controlled, reproducible evaluation.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed) if seed is not None else random

    def set_seed(self, seed: int):
        self.rng = random.Random(seed)

    def reset(self, sim: Simulation):
        p = sim.pitch
        red = sim.red_team[0]
        blue = sim.blue_team[0]
        ball = sim.ball
        sim.reset_positions()

        if hasattr(sim.mode, "state"):
            sim.mode.state = "PLAYING"
            sim.mode.kickoff_timer = 0.0

        # Ball spawned in midfield zone
        ball.pos.x = self.rng.uniform(sim.center.x - 120.0, sim.center.x + 120.0)
        ball.pos.y = self.rng.uniform(p.top + 100.0, p.bottom - 100.0)
        ball.vel = Vec2(0.0, 0.0)

        # Red spawned on West arc around ball
        angle_r = self.rng.uniform(0.6 * math.pi, 1.4 * math.pi)
        dist_r = self.rng.uniform(90.0, 220.0)
        red.pos = ball.pos + Vec2(math.cos(angle_r) * dist_r, math.sin(angle_r) * dist_r)
        red.pos.x = max(p.left + red.radius + 10.0, min(p.right - red.radius - 10.0, red.pos.x))
        red.pos.y = max(p.top + red.radius + 10.0, min(p.bottom - red.radius - 10.0, red.pos.y))
        red.vel = Vec2(0.0, 0.0)
        red.kick_cooldown_timer = 0.0

        # Blue spawned on East arc around ball
        angle_b = self.rng.uniform(-0.4 * math.pi, 0.4 * math.pi)
        dist_b = self.rng.uniform(90.0, 220.0)
        blue.pos = ball.pos + Vec2(math.cos(angle_b) * dist_b, math.sin(angle_b) * dist_b)
        blue.pos.x = max(p.left + blue.radius + 10.0, min(p.right - blue.radius - 10.0, blue.pos.x))
        blue.pos.y = max(p.top + blue.radius + 10.0, min(p.bottom - blue.radius - 10.0, blue.pos.y))
        blue.vel = Vec2(0.0, 0.0)
        blue.kick_cooldown_timer = 0.0
