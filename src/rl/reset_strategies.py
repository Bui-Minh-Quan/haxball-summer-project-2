from abc import ABC, abstractmethod
import math
import random
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class BaseResetStrategy(ABC):

    @abstractmethod
    def reset(self, sim: Simulation):
        pass


class ProximalStrikerReset(BaseResetStrategy):
    """
    Stage 1 Core Reset Strategy:
    Spawns the ball in the attacking half and the agent directly behind it.
    Teaches direct approach, alignment, ball contact, and clean shooting.
    """

    def __init__(self):
        pass

    def reset(self, sim: Simulation):
        p = sim.pitch
        agent = sim.red_team[0]
        ball = sim.ball
        sim.reset_positions()

        # 1. Spawn ball in attacking half
        ball.pos.x = random.uniform(sim.center.x, p.right - 250.0)
        ball.pos.y = random.uniform(p.top + 150.0, p.bottom - 150.0)
        ball.vel = Vec2(0.0, 0.0)

        # 2. Spawn agent strictly behind the ball facing the target net
        angle = random.uniform(math.pi * 0.75, math.pi * 1.25)
        dist = random.uniform(40.0, 90.0)
        agent.pos.x = ball.pos.x + math.cos(angle) * dist
        agent.pos.y = ball.pos.y + math.sin(angle) * dist
        agent.vel = Vec2(0.0, 0.0)
        agent.kick_cooldown_timer = 0.0


class MatchKickoffReset(BaseResetStrategy):
    """Standard game reset for match play and future stages."""

    def reset(self, sim: Simulation):
        sim.reset_positions()