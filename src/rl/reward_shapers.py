from abc import ABC, abstractmethod
from typing import Any
from src.engine.simulation import Simulation
from src.engine.vector import Vec2

class BaseRewardShaper(ABC):
    @abstractmethod
    def reset(self, sim: Simulation):
        pass

    @abstractmethod
    def compute_reward(self, sim: Simulation, goal_event: str | None, truncated: bool) -> tuple[float, bool, dict]:
        pass

class DenseReward(BaseRewardShaper):
    """Team-agnostic dense reward for fixed-horizon training."""
    
    def __init__(self, team: str = "red"):
        self.team = team
        self.prev_dist_to_ball = 0.0
        self.prev_ball_to_goal = 0.0

    def _get_agent_and_goal(self, sim: Simulation):
        p = sim.pitch
        if self.team == "red":
            return sim.red_team[0], Vec2(p.right, sim.center.y), 1.0
        return sim.blue_team[0], Vec2(p.left, sim.center.y), -1.0

    def reset(self, sim: Simulation):
        agent, opp_goal, _ = self._get_agent_and_goal(sim)
        self.prev_dist_to_ball = agent.pos.distance_to(sim.ball.pos)
        self.prev_ball_to_goal = sim.ball.pos.distance_to(opp_goal)

    def compute_reward(self, sim: Simulation, goal_event: str | None, truncated: bool) -> tuple[float, bool, dict]:
        agent, opp_goal, sign = self._get_agent_and_goal(sim)
        ball = sim.ball

        curr_dist_to_ball = agent.pos.distance_to(ball.pos)
        curr_ball_to_goal = ball.pos.distance_to(opp_goal)

        reward = -0.05  # Mild time pressure

        # 1. Active Gradients
        reward += (self.prev_dist_to_ball - curr_dist_to_ball) * 0.03
        reward += (self.prev_ball_to_goal - curr_ball_to_goal) * 0.05

        # 2. Touch Bonus (Directionally aware based on team)
        touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 2.0
        if curr_dist_to_ball <= touch_reach and agent.is_kicking and (ball.vel.x * sign) > 150.0:
            reward += 1.0

        # 3. Match Events
        if goal_event == f"{self.team}_goal":
            reward += 100.0
        elif goal_event is not None:
            reward -= 100.0

        self.prev_dist_to_ball = curr_dist_to_ball
        self.prev_ball_to_goal = curr_ball_to_goal

        info = {"is_goal": goal_event == f"{self.team}_goal", "goal_event": goal_event}
        return reward, False, info  # Terminated is False for fixed-horizon