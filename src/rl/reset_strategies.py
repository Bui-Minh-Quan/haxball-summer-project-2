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

class ActiveSparringReset(BaseResetStrategy):
    """
    Randomized Scramble Possession Reset:
    Generates varied 1v1 tactical scenarios across the pitch,
    bypassing rigid kickoff walls to preserve fluid attacking mechanics.
    """

    def __init__(self, kickoff_prob: float = 0.15):
        self.kickoff_prob = kickoff_prob

    def reset(self, sim: Simulation):
        p = sim.pitch
        agent = sim.red_team[0]
        bot = sim.blue_team[0]
        ball = sim.ball
        sim.reset_positions()

        if random.random() < self.kickoff_prob:
            # Match Kickoff scenario
            if hasattr(sim.mode, "state"):
                sim.mode.state = "KICKOFF"
                sim.mode.kickoff_team = "red" if random.random() < 0.6 else "blue"
                sim.mode.kickoff_timer = 0.0
        else:
            # Active Open-Play Contest
            if hasattr(sim.mode, "state"):
                sim.mode.state = "PLAYING"
                sim.mode.kickoff_timer = 0.0

            # Ball spawns in midfield / central area
            ball.pos.x = random.uniform(sim.center.x - 200.0, sim.center.x + 200.0)
            ball.pos.y = random.uniform(p.top + 100.0, p.bottom - 100.0)
            ball.vel = Vec2(0.0, 0.0)

            # Agent spawns on attacking or defensive side of the ball
            agent_x = max(p.left + 80.0, min(p.right - 80.0, ball.pos.x - random.uniform(60.0, 180.0)))
            agent_y = random.uniform(p.top + 80.0, p.bottom - 80.0)
            agent.pos = Vec2(agent_x, agent_y)
            agent.vel = Vec2(0.0, 0.0)
            agent.kick_cooldown_timer = 0.0

            # Bot spawns on the opposing side
            bot_x = max(p.left + 80.0, min(p.right - 80.0, ball.pos.x + random.uniform(60.0, 220.0)))
            bot_y = random.uniform(p.top + 80.0, p.bottom - 80.0)
            bot.pos = Vec2(bot_x, bot_y)
            bot.vel = Vec2(0.0, 0.0)
            bot.kick_cooldown_timer = 0.0

