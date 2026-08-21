from abc import ABC, abstractmethod
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from typing import Any


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




class Stage2Reward(BaseRewardShaper):
    """
    Reward Shaper for Stage 2 (Continuous Play).
    Features: No truncation on goals, Kickoff penalties, Safe Dribbling, Shot-on-Target rays.
    """

    def __init__(self, team: str = "red"):
        self.team = team
        self.prev_ball_dist_to_opp_goal = 0.0
        self.prev_agent_dist_to_ball = 0.0
        self.shot_cooldown = 0
        self.episode_touched = False

    def reset(self, sim: Any):
        p = sim.pitch
        agent = sim.red_team[0] if self.team == "red" else sim.blue_team[0]
        ball = sim.ball
        opp_goal = Vec2(p.right if self.team == "red" else p.left, sim.center.y)

        self.prev_ball_dist_to_opp_goal = ball.pos.distance_to(opp_goal)
        self.prev_agent_dist_to_ball = agent.pos.distance_to(ball.pos)
        self.shot_cooldown = 0
        self.episode_touched = False

    def compute_reward(self, sim: Any, goal_event: str | None, truncated: bool) -> tuple[float, bool, dict]:
        reward = 0.0
        terminated = False  # NEVER terminate on goal, allowing continuous play
        
        p = sim.pitch
        sign = 1.0 if self.team == "red" else -1.0
        agent = sim.red_team[0] if self.team == "red" else sim.blue_team[0]
        opp_team = sim.blue_team if self.team == "red" else sim.red_team
        ball = sim.ball

        opp_goal = Vec2(p.right if self.team == "red" else p.left, sim.center.y)
        own_goal = Vec2(p.left if self.team == "red" else p.right, sim.center.y)

        curr_ball_dist_to_opp_goal = ball.pos.distance_to(opp_goal)
        curr_agent_dist_to_ball = agent.pos.distance_to(ball.pos)

        # 1. Sparse Goal / Concede Events
        is_my_goal = (goal_event == f"{self.team}_goal")
        is_opp_goal = (goal_event == ("blue_goal" if self.team == "red" else "red_goal"))

        if is_my_goal:
            reward += 100.0
        elif is_opp_goal:
            reward -= 100.0

        # 2. Kick-Off Stalling Penalty
        if getattr(sim.mode, "state", "") == "KICKOFF" and getattr(sim.mode, "kickoff_team", "") == self.team:
            reward -= 0.05  # -3.0 total penalty per second to force immediate play

        # 3. Defensive Screening (Standing between ball and net)
        agent_x, ball_x = agent.pos.x * sign, ball.pos.x * sign
        is_between = agent_x < ball_x
        y_align = abs(agent.pos.y - ball.pos.y)
        if is_between and y_align < 40.0 and curr_agent_dist_to_ball > 60.0:
            reward += 0.02  # Solid positioning bonus

        # 4. Safe Dribbling (Advancing ball with no opponents nearby)
        nearest_opp_dist = min([opp.pos.distance_to(ball.pos) for opp in opp_team]) if opp_team else 999.0
        ball_fwd_delta = (self.prev_ball_dist_to_opp_goal - curr_ball_dist_to_opp_goal)
        
        if nearest_opp_dist > 60.0 and curr_agent_dist_to_ball < 40.0:
            reward += ball_fwd_delta * 0.15  # Big reward for exploiting open space
        else:
            reward += ball_fwd_delta * 0.05  # Normal progression reward

        # 5. Shot-on-Target Micro-Reward
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
            
        kick_reach = agent.radius + ball.radius + agent.stats.kick_margin + 6.0
        if agent.is_kicking and curr_agent_dist_to_ball <= kick_reach and self.shot_cooldown == 0:
            # Raycast from ball in the direction of the kick
            kick_dir = (ball.pos - agent.pos).normalize()
            # Distance from ball X to goal line X
            dist_x = opp_goal.x - ball.pos.x
            
            # If kicked forward
            if (dist_x * sign) > 0 and kick_dir.x != 0:
                t = dist_x / kick_dir.x
                intersect_y = ball.pos.y + t * kick_dir.y
                
                # If the trajectory intersects the goal mouth
                if p.goal_top < intersect_y < p.goal_bottom:
                    reward += 10.0
                    self.shot_cooldown = 60  # Prevent spamming reward

        # Step penalty to encourage speed
        reward -= 0.01

        self.prev_ball_dist_to_opp_goal = curr_ball_dist_to_opp_goal
        self.prev_agent_dist_to_ball = curr_agent_dist_to_ball

        info = {
            "is_goal": is_my_goal,
            "conceded": is_opp_goal,
            "touched": curr_agent_dist_to_ball < kick_reach,
        }

        return reward, terminated, info