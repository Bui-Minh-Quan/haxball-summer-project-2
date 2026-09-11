import math
from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class TeamHeuristicCoordinator:
  """Role-locked tactical multi-agent coordinator:

  - Fixed non-swapping roles (GK, Sweeper, Striker)
  - Predictive trajectory interception (leads the ball)
  - Open-corner post sniping (corners rather than net center)
  - Defensive half boundary clamping for GK
  - Targeted clearance passing to open strikers
  """

  def __init__(self, team: str = "blue"):
    self.team = team

  def _is_path_blocked(
      self,
      start: Vec2,
      end: Vec2,
      opponents: list[Player],
      radius_threshold: float = 38.0,
  ) -> bool:
    """Raycast check to see if an opponent obstructs a shot or passing trajectory."""
    ray = end - start
    ray_len = ray.length()
    if ray_len < 1e-4:
      return False

    ray_dir = ray.normalize()
    for opp in opponents:
      to_opp = opp.pos - start
      proj = to_opp.x * ray_dir.x + to_opp.y * ray_dir.y
      if 0.0 < proj < ray_len:
        perp_dist = abs(to_opp.x * (-ray_dir.y) + to_opp.y * ray_dir.x)
        if perp_dist < radius_threshold:
          return True
    return False

  def _determine_role(self, bot_player: Player, my_team: list[Player]) -> str:
    """Assigns deterministic, non-swapping tactical roles by roster index."""
    if len(my_team) == 1:
      return "STRIKER"

    try:
      idx = my_team.index(bot_player)
    except ValueError:
      return "STRIKER"

    if len(my_team) == 2:
      return "GK" if idx == 0 else "STRIKER"
    else:
      # 3v3 setup: 1 GK, 1 Sweeper/Defender, 1 Striker
      if idx == 0:
        return "GK"
      elif idx == 1:
        return "DEFENDER"
      return "STRIKER"

  def _select_shot_target(
      self,
      pred_ball: Vec2,
      opp_team: list[Player],
      opp_goal_x: float,
      center_y: float,
      goal_h: float,
  ) -> Vec2:
    """Selects the most vulnerable goal post corner furthest from the opponent keeper."""
    post_margin = 18.0
    top_post = Vec2(opp_goal_x, center_y - (goal_h * 0.5) + post_margin)
    bot_post = Vec2(opp_goal_x, center_y + (goal_h * 0.5) - post_margin)

    if not opp_team:
      return Vec2(opp_goal_x, center_y)

    # Find opponent closest to the goal line (the active keeper)
    opp_gk = min(
        opp_team, key=lambda opp: abs(opp.pos.x - opp_goal_x)
    )

    # Aim for the post furthest from the opponent keeper's vertical position
    dist_to_top = opp_gk.pos.distance_to(top_post)
    dist_to_bot = opp_gk.pos.distance_to(bot_post)

    # Prioritize unobstructed post
    top_blocked = self._is_path_blocked(pred_ball, top_post, opp_team)
    bot_blocked = self._is_path_blocked(pred_ball, bot_post, opp_team)

    if not top_blocked and bot_blocked:
      return top_post
    if not bot_blocked and top_blocked:
      return bot_post

    return top_post if dist_to_top >= dist_to_bot else bot_post

  def get_action(
      self, bot_player: Player, sim: Simulation
  ) -> tuple[Vec2, bool]:
    my_team = sim.red_team if self.team == "red" else sim.blue_team
    opp_team = sim.blue_team if self.team == "red" else sim.red_team
    ball = sim.ball
    p = sim.pitch
    sign = 1.0 if self.team == "red" else -1.0

    pitch_w = p.right - p.left
    pitch_h = p.bottom - p.top
    goal_h = getattr(p, "goal_height", 220.0)

    own_goal_x = p.left if self.team == "red" else p.right
    opp_goal_x = p.right if self.team == "red" else p.left
    own_goal_pos = Vec2(own_goal_x, sim.center.y)

    role = self._determine_role(bot_player, my_team)
    safe_m = bot_player.radius + 8.0

    # ── 1. Ball Prediction (Interception Vector) ──
    dist_to_ball = bot_player.pos.distance_to(ball.pos)
    # Lead time scales dynamically with distance to anticipate ball velocity
    lead_time = max(0.04, min(0.32, dist_to_ball / 750.0))
    pred_ball_pos = ball.pos + (ball.vel * lead_time)

    # ── 2. Strategy by Role ──
    target = bot_player.pos
    kick = False
    kick_reach = (
        bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0
    )

    # ==========================================================
    # ROLE: STRIKER
    # ==========================================================
    if role == "STRIKER":
      shot_target = self._select_shot_target(
          pred_ball_pos, opp_team, opp_goal_x, sim.center.y, goal_h
      )
      shot_dir = (shot_target - pred_ball_pos).normalize()

      # Offset approach position (directly behind ball relative to chosen post)
      ideal_strike_pos = pred_ball_pos - (shot_dir * (bot_player.radius + 14.0))

      to_ball = pred_ball_pos - bot_player.pos
      dist_ball = to_ball.length()
      dir_to_ball = to_ball.normalize() if dist_ball > 0 else shot_dir

      # Alignment: 1.0 = bot is in direct firing position behind ball
      alignment = (
          dir_to_ball.x * shot_dir.x + dir_to_ball.y * shot_dir.y
      )

      if alignment > 0.40:
        # CHARGE: Drive straight through the ball into the corner
        target = pred_ball_pos + (shot_dir * 30.0)
      else:
        # ARC AROUND: Curve around ball to prevent knocking it backward
        cross = shot_dir.x * to_ball.y - shot_dir.y * to_ball.x
        side = 1.0 if cross >= 0 else -1.0
        perp = Vec2(-shot_dir.y * side, shot_dir.x * side)

        evasion_weight = max(0.0, 1.0 - (dist_ball / 130.0))
        target = ideal_strike_pos + (perp * 80.0 * evasion_weight)

      # Striker Kick Timing
      if bot_player.pos.distance_to(ball.pos) <= kick_reach:
        bot_to_ball = (ball.pos - bot_player.pos).normalize()
        # Ensure kick drives ball forward toward opponent half
        if (bot_to_ball.x * sign) > 0.15:
          shot_alignment = (
              bot_to_ball.x * shot_dir.x + bot_to_ball.y * shot_dir.y
          )
          if shot_alignment > 0.35:
            kick = True
          # Rebound blast if within shooting distance
          elif ball.pos.distance_to(Vec2(opp_goal_x, sim.center.y)) < (
              pitch_w * 0.30
          ):
            kick = True

    # ==========================================================
    # ROLE: GOALKEEPER (Strict Defensive Half Lockdown)
    # ==========================================================
    elif role == "GK":
      # Defends within defensive half: x restricted between goal line and 30% pitch depth
      max_gk_advance = min(220.0, pitch_w * 0.25)
      gk_base_x = own_goal_x + (sign * 60.0)

      ball_in_defensive_box = (
          sign * (ball.pos.x - own_goal_x) < (pitch_w * 0.22)
          and abs(ball.pos.y - sim.center.y) < (goal_h * 1.2)
      )

      if ball_in_defensive_box:
        # Step out to actively challenge and clear loose balls in the box
        target = ball.pos
      else:
        # Guard goal line: track ball Y clamped inside goal width
        to_ball = ball.pos - own_goal_pos
        step_out = min(
            max_gk_advance, max(45.0, to_ball.length() * 0.18)
        )
        target_x = own_goal_x + (sign * step_out)

        goal_half = (goal_h * 0.5) - 10.0
        target_y = min(
            max(sim.center.y - goal_half, ball.pos.y), sim.center.y + goal_half
        )
        target = Vec2(target_x, target_y)

      # Hard constraint: GK never crosses defensive line
      max_x_allowed = sim.center.x - (sign * (pitch_w * 0.15))
      if sign > 0:
        target.x = min(target.x, max_x_allowed)
      else:
        target.x = max(target.x, max_x_allowed)

      # Clearance / Passing Logic
      if bot_player.pos.distance_to(ball.pos) <= kick_reach:
        bot_to_ball = (ball.pos - bot_player.pos).normalize()
        # Only kick forward into open field
        if (bot_to_ball.x * sign) > -0.1:
          kick = True

    # ==========================================================
    # ROLE: DEFENDER / SWEEPER (3v3 Midfield Anchor)
    # ==========================================================
    elif role == "DEFENDER":
      # Holds space between GK and midfield line
      ball_in_own_half = sign * (ball.pos.x - sim.center.x) < 0

      if ball_in_own_half:
        # Challenge loose balls in defensive midfield
        target = pred_ball_pos - (
            Vec2(sign, 0.0) * (bot_player.radius + 15.0)
        )
      else:
        # Hold midfield anchor position to cut counter-attacks
        anchor_x = sim.center.x - (sign * (pitch_w * 0.14))
        # Mirror ball Y slightly to support attack lane
        anchor_y = sim.center.y + (ball.pos.y - sim.center.y) * 0.45
        target = Vec2(anchor_x, anchor_y)

      # Clamp defender: never push beyond opponent 35% line
      max_def_x = sim.center.x + (sign * (pitch_w * 0.15))
      if sign > 0:
        target.x = min(target.x, max_def_x)
      else:
        target.x = max(target.x, max_def_x)

      # Clearance kick forward
      if bot_player.pos.distance_to(ball.pos) <= kick_reach:
        bot_to_ball = (ball.pos - bot_player.pos).normalize()
        if (bot_to_ball.x * sign) > 0.0:
          kick = True

    # ── 3. Pitch Boundary Clamping & Motor Output ──
    target.x = max(
        p.outer_left + safe_m, min(p.outer_right - safe_m, target.x)
    )
    target.y = max(
        p.outer_top + safe_m, min(p.outer_bottom - safe_m, target.y)
    )

    to_target = target - bot_player.pos
    dist_target = to_target.length()
    move_dir = (
        to_target.normalize() if dist_target > 6.0 else Vec2(0.0, 0.0)
    )

    return move_dir, kick