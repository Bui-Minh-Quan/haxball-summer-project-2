import math
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
  """1.

  First-touch reward: Granted strictly once per kickoff/reset to initiate
  play. 2. Continuous Ball-to-Goal penalty: Ranges from 0.0 at the goal line
  down to -0.05 at opposite net. 3. Terminal Match Events: +100 for scored goal,
  -100 for conceded goal.
  """

  def __init__(
      self,
      team: str = "red",
      first_touch_bonus: float = 10.0,
      ball_penalty_weight: float = 0.05,
  ):
    self.team = team
    self.first_touch_bonus = first_touch_bonus
    self.ball_penalty_weight = ball_penalty_weight
    self.has_touched_ball = False

  def _get_entities_and_goal(self, sim: Simulation):
    p = sim.pitch
    is_red = self.team == "red"
    agent = sim.red_team[0] if is_red else sim.blue_team[0]
    sign = 1.0 if is_red else -1.0

    opp_goal_x = p.right if is_red else p.left
    goal_top = p.goal_top
    goal_bottom = p.goal_bottom

    return agent, opp_goal_x, goal_top, goal_bottom, sign

  def _dist_to_goal_segment(
      self, pos: Vec2, goal_x: float, top: float, bottom: float
  ) -> float:
    clamped_y = max(top, min(bottom, pos.y))
    dx = pos.x - goal_x
    dy = pos.y - clamped_y
    return math.hypot(dx, dy)

  def reset(self, sim: Simulation):
    self.has_touched_ball = False

  def compute_reward(
      self, sim: Simulation, goal_event: str | None, truncated: bool
  ) -> tuple[float, bool, dict]:
    agent, opp_goal_x, goal_top, goal_bottom, _ = (
        self._get_entities_and_goal(sim)
    )
    ball = sim.ball
    p = sim.pitch
    max_pitch_diag = math.hypot(p.width, p.height)

    # 1. Distances
    dist_player_to_ball = agent.pos.distance_to(ball.pos)
    dist_ball_to_goal = self._dist_to_goal_segment(
        ball.pos, opp_goal_x, goal_top, goal_bottom
    )

    # 2. Continuous Ball-to-Goal Penalty (Closer to net = less penalty)
    # At goal line = 0.0, at opposite end ≈ -0.05 per tick
    ball_penalty = -(dist_ball_to_goal / max_pitch_diag) * self.ball_penalty_weight
    reward = ball_penalty

    # 3. First Touch Bonus (Triggered once per reset/kickoff)
    touch_reach = agent.radius + ball.radius + agent.stats.kick_margin + 4.0
    if not self.has_touched_ball and dist_player_to_ball <= touch_reach:
      reward += self.first_touch_bonus
      self.has_touched_ball = True

    # 4. Terminal Match Goals
    if goal_event == f"{self.team}_goal":
      reward += 100.0
      self.has_touched_ball = False
    elif goal_event is not None:
      reward -= 100.0
      self.has_touched_ball = False

    info = {
        "dist_player_ball": dist_player_to_ball,
        "dist_ball_goal": dist_ball_to_goal,
        "has_touched": self.has_touched_ball,
        "goal_event": goal_event,
    }
    return reward, False, info


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