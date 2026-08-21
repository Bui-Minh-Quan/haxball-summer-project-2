import math
import random
from abc import ABC, abstractmethod
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class BaseResetStrategy(ABC):

    @abstractmethod
    def reset(self, sim: Simulation):
        pass


class RandomPitchReset(BaseResetStrategy):
    """
    Balanced Full-Pitch Reset Strategy:
    Mixes close-range alignment scenarios (50%) with full-pitch navigation (50%)
    to maintain a consistent gradient from initial contact to goal scoring.
    """

    def __init__(self, min_distance: float = 60.0):
        self.min_distance = min_distance

    def reset(self, sim: Simulation):
        p = sim.pitch
        agent = sim.red_team[0]
        ball = sim.ball
        sim.reset_positions()

        opp_goal = Vec2(p.right, sim.center.y)

        # 50% Focused Striking Spawn / 50% Wide Open Pitch Exploration
        if random.random() < 0.50:
            # Ball in attacking half
            ball.pos.x = random.uniform(sim.center.x - 100.0, p.right - 220.0)
            ball.pos.y = random.uniform(p.top + 100.0, p.bottom - 100.0)
            ball.vel = Vec2(0.0, 0.0)

            # Agent within 150px of the ball at any angle (forces orbiting)
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(self.min_distance, 150.0)
            agent.pos.x = max(p.left + 50.0, min(p.right - 50.0, ball.pos.x + math.cos(angle) * dist))
            agent.pos.y = max(p.top + 50.0, min(p.bottom - 50.0, ball.pos.y + math.sin(angle) * dist))
        else:
            # Full Pitch Random Spawns
            ball.pos.x = random.uniform(p.left + 150.0, p.right - 220.0)
            ball.pos.y = random.uniform(p.top + 80.0, p.bottom - 80.0)
            ball.vel = Vec2(0.0, 0.0)

            while True:
                ax = random.uniform(p.left + 80.0, p.right - 80.0)
                ay = random.uniform(p.top + 60.0, p.bottom - 60.0)
                if Vec2(ax, ay).distance_to(ball.pos) >= self.min_distance:
                    agent.pos = Vec2(ax, ay)
                    break

        agent.vel = Vec2(0.0, 0.0)
        agent.kick_cooldown_timer = 0.0


class MatchKickoffReset(BaseResetStrategy):

    def reset(self, sim: Simulation):
        sim.reset_positions()