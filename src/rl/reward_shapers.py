from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class BaseRewardShaper(ABC):
    @abstractmethod
    def reset(self, sim: Simulation):
        pass

    @abstractmethod
    def compute_reward(
        self, sim: Simulation, goal_event: str | None, truncated: bool
    ) -> tuple[float, bool, dict]:
        pass


class DenseReward(BaseRewardShaper):
    """Gated reward: Rewards ball pursuit without penalizing the agent when the ball is kicked away."""

    def __init__(self, team: str = "red"):
        self.team = team
        self.prev_dist_to_ball = 0.0
        self.prev_ball_to_goal = 0.0

    def _get_entities(self, sim: Simulation):
        p = sim.pitch
        if self.team == "red":
            agent = sim.red_team[0]
            opp_goal = Vec2(p.right, sim.center.y)
            own_goal = Vec2(p.left, sim.center.y)
            sign = 1.0
        else:
            agent = sim.blue_team[0]
            opp_goal = Vec2(p.left, sim.center.y)
            own_goal = Vec2(p.right, sim.center.y)
            sign = -1.0
        return agent, opp_goal, own_goal, sign

    def reset(self, sim: Simulation):
        agent, opp_goal, _, _ = self._get_entities(sim)
        self.prev_dist_to_ball = agent.pos.distance_to(sim.ball.pos)
        self.prev_ball_to_goal = sim.ball.pos.distance_to(opp_goal)

    def compute_reward(
        self, sim: Simulation, goal_event: str | None, truncated: bool
    ) -> tuple[float, bool, dict]:
        agent, opp_goal, own_goal, sign = self._get_entities(sim)
        ball = sim.ball

        curr_dist_to_ball = agent.pos.distance_to(ball.pos)
        curr_ball_to_goal = ball.pos.distance_to(opp_goal)

        reward = -0.1  # Minimal alive penalty

        # --- 1. Ball Progression toward Opponent Goal ---
        # Ball moving toward target net always yields positive reward
        ball_progress = (self.prev_ball_to_goal - curr_ball_to_goal)
        reward += ball_progress * 0.08

        # --- 2. Gated Player-to-Ball Pursuit ---
        # ONLY reward/penalize agent distance if the ball is NOT actively flying toward the goal
        ball_moving_to_target = (ball.vel.x * sign) > 50.0
        if not ball_moving_to_target:
            dist_delta = self.prev_dist_to_ball - curr_dist_to_ball
            # Clip negative deltas so agent is never punished when ball bounces away
            reward += max(dist_delta, -2.0) * 0.04

        # --- 3. Directional Strike Bonus ---
        touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 4.0
        if curr_dist_to_ball <= touch_reach:
            # Reward kicking forward
            if agent.is_kicking and (ball.vel.x * sign) > 100.0:
                reward += 2.0
            # Small touch bonus to encourage making physical contact
            elif not agent.is_kicking:
                reward += 0.1

        # --- 4. Match Terminal Events ---
        if goal_event == f"{self.team}_goal":
            reward += 100.0
        elif goal_event is not None:
            reward -= 100.0

        self.prev_dist_to_ball = curr_dist_to_ball
        self.prev_ball_to_goal = curr_ball_to_goal

        return reward, False, {"goal_event": goal_event}

class BallChaserReward(BaseRewardShaper):
    """Sanity Check Shaper: Strictly rewards chasing, touching, and kicking the ball."""

    def __init__(self, team: str = "red"):
        self.team = team
        self.prev_dist_to_ball = 0.0

    def _get_agent(self, sim: Simulation):
        return sim.red_team[0] if self.team == "red" else sim.blue_team[0]

    def reset(self, sim: Simulation):
        agent = self._get_agent(sim)
        self.prev_dist_to_ball = agent.pos.distance_to(sim.ball.pos)

    def compute_reward(
        self, sim: Simulation, goal_event: str | None, truncated: bool
    ) -> tuple[float, bool, dict]:
        agent = self._get_agent(sim)
        ball = sim.ball
        curr_dist = agent.pos.distance_to(ball.pos)

        # 1. Continuous approach gradient (+0.1 per pixel closed)
        dist_delta = self.prev_dist_to_ball - curr_dist
        reward = dist_delta * 0.1

        # 2. Contact bonus (Huge reward for physical proximity)
        touch_margin = agent.radius + ball.radius + agent.stats.kick_margin + 5.0
        is_touching = curr_dist <= touch_margin

        if is_touching:
            reward += 1.0  # +1.0 for every step touching/near the ball

        # 3. Kick bonus while in contact
        if is_touching and agent.is_kicking:
            reward += 2.0

        self.prev_dist_to_ball = curr_dist

        info = {
            "dist_to_ball": curr_dist,
            "is_touching": is_touching,
            "is_kicking": agent.is_kicking,
        }
        return reward, False, info