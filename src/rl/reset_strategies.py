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

                # Spawn ball centrally
            ball.pos.x = random.uniform(p.left + 250.0, p.right - 250.0)
            ball.pos.y = random.uniform(p.top + 100.0, p.bottom - 100.0)
            ball.vel = Vec2(0.0, 0.0)

            # Randomize agent position in an arc around the ball
            angle_agent = random.uniform(0.6 * math.pi, 1.4 * math.pi)  # Western arc
            dist_agent = random.uniform(80.0, 220.0)
            agent.pos = ball.pos + Vec2(
                math.cos(angle_agent) * dist_agent,
                math.sin(angle_agent) * dist_agent,
            )

            # Randomize bot position in an independent arc
            angle_bot = random.uniform(-0.4 * math.pi, 0.4 * math.pi)  # Eastern arc
            dist_bot = random.uniform(80.0, 220.0)
            bot.pos = ball.pos + Vec2(
                math.cos(angle_bot) * dist_bot, math.sin(angle_bot) * dist_bot
            )
            bot.vel = Vec2(0.0, 0.0)
            bot.kick_cooldown_timer = 0.0


class MultiAgentScrambleReset(BaseResetStrategy):
    """Fully Symmetric Multi-Agent Reset.

    Randomizes attacking and defending starting arcs evenly between Red and Blue.
    """

    def reset(self, sim: Simulation):
        p = sim.pitch
        ball = sim.ball
        sim.reset_positions()

        if hasattr(sim.mode, "state"):
            sim.mode.state = "PLAYING"
            sim.mode.kickoff_timer = 0.0

        # Spawn ball in central pitch
        ball.pos.x = random.uniform(p.left + 300.0, p.right - 300.0)
        ball.pos.y = random.uniform(p.top + 150.0, p.bottom - 150.0)
        ball.vel = Vec2(0.0, 0.0)

        # 50% chance Red is west/attacker, 50% chance Blue is west/attacker
        red_is_west = random.random() < 0.5

        angle_west = random.uniform(0.6 * math.pi, 1.4 * math.pi)
        angle_east = random.uniform(-0.4 * math.pi, 0.4 * math.pi)

        dist_red = random.uniform(90.0, 260.0)
        dist_blue = random.uniform(90.0, 260.0)

        ang_r = angle_west if red_is_west else angle_east
        ang_b = angle_east if red_is_west else angle_west

        for player in sim.red_team:
            pos = ball.pos + Vec2(
                math.cos(ang_r) * dist_red, math.sin(ang_r) * dist_red
            )
            player.pos.x = max(
                p.left + player.radius, min(p.right - player.radius, pos.x)
            )
            player.pos.y = max(
                p.top + player.radius, min(p.bottom - player.radius, pos.y)
            )
            player.vel = Vec2(0.0, 0.0)
            player.kick_cooldown_timer = 0.0

        for player in sim.blue_team:
            pos = ball.pos + Vec2(
                math.cos(ang_b) * dist_blue, math.sin(ang_b) * dist_blue
            )
            player.pos.x = max(
                p.left + player.radius, min(p.right - player.radius, pos.x)
            )
            player.pos.y = max(
                p.top + player.radius, min(p.bottom - player.radius, pos.y)
            )
            player.vel = Vec2(0.0, 0.0)
            player.kick_cooldown_timer = 0.0


class FixedMultiAgentReset(BaseResetStrategy):
    """Deterministic, mirror-symmetric reset for self-play.

    - Ball is placed dead center.
    - Red and Blue players spawn at identical mirrored offsets.
    - Works for 1v1 up to N vs M rosters.
    """

    def __init__(self, offset_x: float = 180.0):
        self.offset_x = offset_x

    def reset(self, sim: Simulation):
        p = sim.pitch
        sim.reset_positions()

        if hasattr(sim.mode, "state"):
            sim.mode.state = "PLAYING"
            sim.mode.kickoff_timer = 0.0

        # 1. Ball at dead center
        sim.ball.pos = sim.center.copy()
        sim.ball.vel = Vec2(0.0, 0.0)

        # 2. Red Team (Left side)
        num_red = len(sim.red_team)
        y_step_red = p.height / (num_red + 1)
        for i, player in enumerate(sim.red_team):
            player.pos = Vec2(
                sim.center.x - self.offset_x, p.top + y_step_red * (i + 1)
            )
            player.vel = Vec2(0.0, 0.0)
            player.kick_cooldown_timer = 0.0

        # 3. Blue Team (Right side, exact mirror)
        num_blue = len(sim.blue_team)
        y_step_blue = p.height / (num_blue + 1)
        for j, player in enumerate(sim.blue_team):
            player.pos = Vec2(
                sim.center.x + self.offset_x, p.top + y_step_blue * (j + 1)
            )
            player.vel = Vec2(0.0, 0.0)
            player.kick_cooldown_timer = 0.0

class TwoVTwoScrambleReset(BaseResetStrategy):
    """Spawns 1 ST and 1 CB per team in tactically relevant initial zones."""
    
    def reset(self, sim: Simulation):
        p = sim.pitch
        ball = sim.ball
        sim.reset_positions()

        if hasattr(sim.mode, "state"):
            sim.mode.state = "PLAYING"
            sim.mode.kickoff_timer = 0.0

        # Ball in midfield
        ball.pos.x = random.uniform(sim.center.x - 150.0, sim.center.x + 150.0)
        ball.pos.y = random.uniform(p.top + 150.0, p.bottom - 150.0)
        ball.vel = Vec2(0.0, 0.0)

        # RED TEAM
        red_st = sim.red_team[0]
        red_cb = sim.red_team[1]
        
        # Red ST spawns near center line
        red_st.pos = Vec2(sim.center.x - random.uniform(50.0, 200.0), random.uniform(p.top+100, p.bottom-100))
        # Red CB spawns deep
        red_cb.pos = Vec2(p.left + random.uniform(100.0, 250.0), random.uniform(p.top+100, p.bottom-100))

        # BLUE TEAM
        blue_st = sim.blue_team[0]
        blue_cb = sim.blue_team[1]
        
        # Blue ST spawns near center line
        blue_st.pos = Vec2(sim.center.x + random.uniform(50.0, 200.0), random.uniform(p.top+100, p.bottom-100))
        # Blue CB spawns deep
        blue_cb.pos = Vec2(p.right - random.uniform(100.0, 250.0), random.uniform(p.top+100, p.bottom-100))

        for player in sim.all_players:
            player.vel = Vec2(0.0, 0.0)
            player.kick_cooldown_timer = 0.0