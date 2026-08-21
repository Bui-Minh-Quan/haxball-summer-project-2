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
    """
    Stage 1 Full-Pitch Potential Reward:
    - Orbit & Approach Potential (navigates behind the ball)
    - Ball Progression Potential (drives ball to target goal)
    - Clean Strike Bonus (rewards forward velocity on contact)
    """

    def __init__(self):
        self.prev_dist_to_prep = 0.0
        self.prev_ball_to_goal = 0.0
        self.has_scored = False
        self.touched_this_ep = False

    def reset(self, sim: Simulation):
        p = sim.pitch
        agent = sim.red_team[0]
        ball = sim.ball
        opp_goal = Vec2(p.right, sim.center.y)

        goal_dir = (opp_goal - ball.pos).normalize()
        prep_pos = ball.pos - (goal_dir * 35.0)

        self.prev_dist_to_prep = agent.pos.distance_to(prep_pos)
        self.prev_ball_to_goal = ball.pos.distance_to(opp_goal)
        self.has_scored = False
        self.touched_this_ep = False

    def compute_reward(
        self, sim: Simulation, goal_event: str | None, truncated: bool
    ) -> tuple[float, bool, dict]:
        agent = sim.red_team[0]
        ball = sim.ball
        p = sim.pitch
        opp_goal = Vec2(p.right, sim.center.y)

        goal_dir = (opp_goal - ball.pos).normalize()
        prep_pos = ball.pos - (goal_dir * 35.0)

        curr_dist_to_prep = agent.pos.distance_to(prep_pos)
        curr_ball_to_goal = ball.pos.distance_to(opp_goal)
        dist_to_ball = agent.pos.distance_to(ball.pos)
        touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 2.0

        reward = -0.05  # Lightweight step cost

        # 1. Orbit & Approach Gradient (Continuous guidance to get behind the ball)
        prep_delta = self.prev_dist_to_prep - curr_dist_to_prep
        reward += prep_delta * 0.04

        # 2. Ball Progress Gradient (Positive when ball moves closer to target net)
        ball_delta = self.prev_ball_to_goal - curr_ball_to_goal
        reward += ball_delta * 0.08

        # 3. Direct Contact & Forward Shot Bonus
        if dist_to_ball <= touch_reach:
            self.touched_this_ep = True
            # Reward kicking/pushing ball forward
            if ball.vel.x > 50.0:
                reward += (ball.vel.x / 1000.0) * 0.1

        # 4. Terminal Rewards
        terminated = False
        if goal_event == "red_goal":
            reward += 100.0
            self.has_scored = True
            terminated = True
        elif goal_event == "blue_goal":
            reward -= 50.0  # Punish own goals
            terminated = True

        

        self.prev_dist_to_prep = curr_dist_to_prep
        self.prev_ball_to_goal = curr_ball_to_goal

        info = {
            "is_goal": self.has_scored,
            "conceded": goal_event == "blue_goal",
            "touched": self.touched_this_ep,
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