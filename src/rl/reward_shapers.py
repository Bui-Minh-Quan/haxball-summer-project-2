from abc import ABC, abstractmethod
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from typing import Any
import numpy as np


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


class Stage2ScratchReward(BaseRewardShaper):
    """
    Scratch 1v1 Sparring Reward:
    - Distance potential to ball approach and goal advancement
    - Milestone for forward strikes
    - Terminal reward on Goal (+100) / Concede (-50)
    """

    def __init__(self, team: str = "red"):
        self.team = team
        self.prev_agent_to_ball = 0.0
        self.prev_ball_to_goal = 0.0
        self.touched_this_ep = False
        self.shot_cooldown = 0

    def reset(self, sim: Any):
        p = sim.pitch
        agent = sim.red_team[0] if self.team == "red" else sim.blue_team[0]
        ball = sim.ball
        opp_goal = Vec2(p.right if self.team == "red" else p.left, sim.center.y)

        self.prev_agent_to_ball = agent.pos.distance_to(ball.pos)
        self.prev_ball_to_goal = ball.pos.distance_to(opp_goal)
        self.touched_this_ep = False
        self.shot_cooldown = 0

    def compute_reward(
        self, sim: Any, goal_event: str | None, truncated: bool
    ) -> tuple[float, bool, dict]:
        p = sim.pitch
        sign = 1.0 if self.team == "red" else -1.0
        agent = sim.red_team[0] if self.team == "red" else sim.blue_team[0]
        ball = sim.ball
        opp_goal = Vec2(p.right if self.team == "red" else p.left, sim.center.y)

        curr_agent_to_ball = agent.pos.distance_to(ball.pos)
        curr_ball_to_goal = ball.pos.distance_to(opp_goal)
        touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 2.0

        reward = -0.01  # Step time cost

        # 1. Approach Guidance (Active until first touch)
        if not self.touched_this_ep:
            agent_delta = self.prev_agent_to_ball - curr_agent_to_ball
            reward += agent_delta * 0.03

        # 2. Ball Progression (Moving ball toward opponent net)
        ball_delta = self.prev_ball_to_goal - curr_ball_to_goal
        reward += ball_delta * 0.05

        # 3. Touch & Strike Bonus
        if curr_agent_to_ball <= touch_reach:
            self.touched_this_ep = True
            if self.shot_cooldown > 0:
                self.shot_cooldown -= 1
            if agent.is_kicking and (ball.vel.x * sign) > 200.0 and self.shot_cooldown == 0:
                reward += 3.0
                self.shot_cooldown = 30

        # 4. Terminal Rewards
        terminated = False
        is_my_goal = goal_event == f"{self.team}_goal"
        is_opp_goal = goal_event == ("blue_goal" if self.team == "red" else "red_goal")

        if is_my_goal:
            reward += 100.0
            terminated = True
        elif is_opp_goal:
            reward -= 50.0
            terminated = True

        self.prev_agent_to_ball = curr_agent_to_ball
        self.prev_ball_to_goal = curr_ball_to_goal

        info = {
            "is_goal": is_my_goal,
            "conceded": is_opp_goal,
            "touched": self.touched_this_ep,
        }

        return reward, terminated, info


class Stage3SelfPlayReward:
    """Strict Zero-Sum Self-Play Reward with Draw Penalties.

    - Score: +100.0
    - Concede: -100.0
    - Timeout / Draw: -50.0 for both agents
    """

    def reset(self, sim: Any):
        pass

    def compute_reward(
        self,
        sim: Any,
        goal_event: str | None,
        truncated: bool,
        rl_slots: list[dict],
    ) -> tuple[np.ndarray, bool, dict]:
        rewards = np.zeros(len(rl_slots), dtype=np.float32)
        terminated = False

        if goal_event is not None:
            terminated = True
            for i, slot in enumerate(rl_slots):
                if goal_event == f"{slot['team']}_goal":
                    rewards[i] += 100.0
                else:
                    rewards[i] -= 100.0
        elif truncated:
            # Active punishment for stalling / running out the clock
            rewards.fill(-70.0)

        info = {
            "is_goal": goal_event is not None,
            "goal_event": goal_event,
            "draw": truncated and (goal_event is None),
        }

        return rewards, terminated, info


class Stage3_2RoleReward:
    """Asymmetric 2v2 Role Rewards (ST + CB)."""

    def __init__(self):
        self.last_touch_team = None
        self.last_touch_player_idx = None
        self.last_touch_pos = None

    def reset(self, sim: Any):
        self.last_touch_team = None
        self.last_touch_player_idx = None
        self.last_touch_pos = None

    def compute_reward(self, sim: Any, goal_event: str | None, truncated: bool, rl_slots: list[dict]) -> tuple:
        rewards = np.zeros(len(rl_slots), dtype=np.float32)
        terminated = False
        ball = sim.ball
        p = sim.pitch
        
        player_to_slot = {slot["player"]: i for i, slot in enumerate(rl_slots)}

        # 1. Event Tracking (Passes, Clearances, Turnovers)
        for player in sim.all_players:
            touch_reach = player.radius + ball.radius + player.stats.kick_margin + 2.0
            
            if player.pos.distance_to(ball.pos) <= touch_reach:
                # Is it an RL agent touching the ball?
                if player in player_to_slot:
                    i = player_to_slot[player]
                    slot = rl_slots[i]
                    sign = 1.0 if slot["team"] == "red" else -1.0
                    
                    is_att_third = (player.pos.x - sim.center.x) * sign > (p.width / 6.0)
                    is_def_third = (player.pos.x - sim.center.x) * sign < -(p.width / 6.0)

                    # A. Striker: High Press Turnover Bonus
                    if slot["role"] == "ST" and is_att_third and self.last_touch_team != slot["team"]:
                        rewards[i] += 1.5  # Won the ball deep in enemy territory
                    
                    # B. Center Back: Clearance Bonus
                    if slot["role"] == "CB" and is_def_third and player.is_kicking:
                        if (ball.vel.x * sign) > 300.0:  # Hard kick away from own net
                            rewards[i] += 1.5
                            
                    # C. Pass Connection
                    if self.last_touch_team == slot["team"] and self.last_touch_player_idx != i:
                        if self.last_touch_pos and self.last_touch_pos.distance_to(ball.pos) > 120.0:
                            # Pass completed! Reward both receiver and passer
                            rewards[i] += 1.0 # Receiver
                            if self.last_touch_player_idx is not None:
                                rewards[self.last_touch_player_idx] += 1.5 # Passer

                # Update state tracking
                self.last_touch_team = "red" if player in sim.red_team else "blue"
                self.last_touch_player_idx = player_to_slot.get(player, None)
                self.last_touch_pos = ball.pos.copy()

        # 2. Terminal Rewards (Asymmetric)
        if goal_event is not None:
            terminated = True
            for i, slot in enumerate(rl_slots):
                is_my_goal = goal_event == f"{slot['team']}_goal"
                
                if slot["role"] == "ST":
                    rewards[i] += 10.0 if is_my_goal else -4.0  # ST rewarded heavily for scoring, mildly punished for conceding
                elif slot["role"] == "CB":
                    rewards[i] += 4.0 if is_my_goal else -10.0  # CB heavily punished for conceding, mildly rewarded for scoring
                    
        elif truncated:
            # Draw timeout
            for i, slot in enumerate(rl_slots):
                if slot["role"] == "ST":
                    rewards[i] -= 3.0  # ST failed to score
                elif slot["role"] == "CB":
                    rewards[i] += 2.0  # CB successfully defended a 0-0 draw!

        info = {"is_goal": goal_event is not None, "goal_event": goal_event}
        return rewards, terminated, info