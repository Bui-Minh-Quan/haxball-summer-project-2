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
    Stage 2 Clean Possession Sparring Reward:
    - Terminated = True on Goal / Concede (Eliminates GAE advantage bleed across possessions)
    - Milestone reward for winning 50-50 physical tackles
    - Dynamic shot creation bonus
    - Continuous orbital positioning guidance
    """

    def __init__(self, team: str = "red"):
        self.team = team
        self.prev_dist_to_prep = 0.0
        self.prev_ball_to_goal = 0.0
        self.prev_agent_to_ball = 0.0
        self.shot_cooldown = 0
        self.touched_this_ep = False

    def reset(self, sim: Any):
        p = sim.pitch
        agent = sim.red_team[0] if self.team == "red" else sim.blue_team[0]
        ball = sim.ball
        opp_goal = Vec2(p.right if self.team == "red" else p.left, sim.center.y)

        goal_dir = (opp_goal - ball.pos).normalize()
        prep_pos = ball.pos - (goal_dir * 35.0)

        self.prev_dist_to_prep = agent.pos.distance_to(prep_pos)
        self.prev_ball_to_goal = ball.pos.distance_to(opp_goal)
        self.prev_agent_to_ball = agent.pos.distance_to(ball.pos)
        self.shot_cooldown = 0
        self.touched_this_ep = False

    def compute_reward(self, sim: Any, goal_event: str | None, truncated: bool) -> tuple[float, bool, dict]:
        p = sim.pitch
        sign = 1.0 if self.team == "red" else -1.0
        agent = sim.red_team[0] if self.team == "red" else sim.blue_team[0]
        opp = (sim.blue_team if self.team == "red" else sim.red_team)[0]
        ball = sim.ball

        opp_goal = Vec2(p.right if self.team == "red" else p.left, sim.center.y)
        goal_dir = (opp_goal - ball.pos).normalize()
        prep_pos = ball.pos - (goal_dir * 35.0)

        curr_dist_to_prep = agent.pos.distance_to(prep_pos)
        curr_ball_to_goal = ball.pos.distance_to(opp_goal)
        curr_agent_to_ball = agent.pos.distance_to(ball.pos)
        opp_to_ball = opp.pos.distance_to(ball.pos)
        touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 2.0

        reward = -0.01  # Mild step time cost

        # 1. Continuous Orbital Guidance (Gets behind the ball)
        prep_delta = self.prev_dist_to_prep - curr_dist_to_prep
        reward += prep_delta * 0.03

        # 2. Ball Progression (Moving ball toward opponent net)
        ball_delta = self.prev_ball_to_goal - curr_ball_to_goal
        reward += ball_delta * 0.05

        # 3. 50-50 Duel Milestone: Win ball when opponent is actively contesting
        if curr_agent_to_ball <= touch_reach:
            if not self.touched_this_ep:
                self.touched_this_ep = True
                if opp_to_ball < 120.0:
                    reward += 3.0  # Big bonus for winning a contested 50-50 challenge

        # 4. Shot Creation Bonus
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1

        if agent.is_kicking and curr_agent_to_ball <= touch_reach and self.shot_cooldown == 0:
            if (ball.vel.x * sign) > 250.0:
                reward += 4.0  # Reward firing a hard shot forward
                self.shot_cooldown = 45

        # 5. Clean Terminal Boundaries (Isolated Possessions)
        terminated = False
        is_my_goal = (goal_event == f"{self.team}_goal")
        is_opp_goal = (goal_event == ("blue_goal" if self.team == "red" else "red_goal"))

        if is_my_goal:
            reward += 100.0
            terminated = True
        elif is_opp_goal:
            reward -= 45.0  # Moderate penalty prevents fear of engaging
            terminated = True

        self.prev_dist_to_prep = curr_dist_to_prep
        self.prev_ball_to_goal = curr_ball_to_goal
        self.prev_agent_to_ball = curr_agent_to_ball

        info = {
            "is_goal": is_my_goal,
            "conceded": is_opp_goal,
            "touched": self.touched_this_ep,
        }

        return reward, terminated, info
    