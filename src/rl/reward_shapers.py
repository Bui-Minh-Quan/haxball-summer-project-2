from abc import ABC, abstractmethod
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


class Stage1Reward(BaseRewardShaper):

    def __init__(self):
        self.prev_ball_to_goal = 0.0
        self.has_touched_ball = False
        self.has_scored = False

    def reset(self, sim: Simulation):
        target_goal = Vec2(sim.pitch.right, sim.center.y)
        self.prev_ball_to_goal = sim.ball.pos.distance_to(target_goal)
        self.has_touched_ball = False
        self.has_scored = False

    def compute_reward(
        self, sim: Simulation, goal_event: str | None, truncated: bool
    ) -> tuple[float, bool, dict]:
        agent = sim.red_team[0]
        ball = sim.ball
        target_goal = Vec2(sim.pitch.right, sim.center.y)

        curr_ball_to_goal = ball.pos.distance_to(target_goal)
        dist_agent_to_ball = agent.pos.distance_to(ball.pos)
        
        # +1.0px tolerance to account for physics engine collision resolution
        touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 1.0

        reward = -0.05  # Step time penalty

        # 1. First Touch Milestone
        if dist_agent_to_ball <= touch_reach:
            if not self.has_touched_ball:
                reward += 5.0
                self.has_touched_ball = True

        # 2. Ball to Goal Progress (Only rewarded after making contact)
        if self.has_touched_ball:
            #print("toched")
            delta_ball = self.prev_ball_to_goal - curr_ball_to_goal
            if delta_ball > 0:
                reward += delta_ball * 0.05  # e.g., moving ball 200px forward = +10.0
        

            
        # 3. Terminal Conditions
        terminated = False
        if goal_event == "red_goal":
            reward += 100.0
            self.has_scored = True
            terminated = True
        elif goal_event == "blue_goal":
            reward -= 100.0  # Punish own goals
            terminated = True

        # 4. Timeout penalty
        if truncated and not terminated:
            reward -= 50.0

        self.prev_ball_to_goal = curr_ball_to_goal

        # Single source of truth for the logger
        info = {
            "is_goal": self.has_scored,
            "touched": self.has_touched_ball,
        }


        return reward, terminated, info