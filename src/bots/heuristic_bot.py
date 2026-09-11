import math
from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class TeamHeuristicCoordinator:
  """Tactical 3-bot coordinator:

  - GK & Defender: Lock down defensive half and pass directly to the striker.
  - Striker: Stays strictly in the attacking half, roaming to open passing lanes.
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
    """Raycast check to determine if an opponent obstructs a trajectory."""
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
    """Assigns deterministic tactical roles by roster index."""
    if len(my_team) == 1:
      return "STRIKER"

    try:
      idx = my_team.index(bot_player)
    except ValueError:
      return "STRIKER"

    if len(my_team) == 2:
      return "GK" if idx == 0 else "STRIKER"
    else:
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
    """Selects the goal post corner furthest from the opponent keeper."""
    post_margin = 18.0
    top_post = Vec2(opp_goal_x, center_y - (goal_h * 0.5) + post_margin)
    bot_post = Vec2(opp_goal_x, center_y + (goal_h * 0.5) - post_margin)

    if not opp_team:
      return Vec2(opp_goal_x, center_y)

    opp_gk = min(opp_team, key=lambda opp: abs(opp.pos.x - opp_goal_x))
    dist_to_top = opp_gk.pos.distance_to(top_post)
    dist_to_bot = opp_gk.pos.distance_to(bot_post)

    top_blocked = self._is_path_blocked(pred_ball, top_post, opp_team)
    bot_blocked = self._is_path_blocked(pred_ball, bot_post, opp_team)

    if not top_blocked and bot_blocked:
      return top_post
    if not bot_blocked and top_blocked:
      return bot_post

    return top_post if dist_to_top >= dist_to_bot else bot_post

  def _find_striker(self, my_team: list[Player]) -> Player | None:
    for pl in my_team:
      if self._determine_role(pl, my_team) == "STRIKER":
        return pl
    return None

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
    striker_teammate = self._find_striker(my_team)
    safe_m = bot_player.radius + 8.0

    # ── 1. Ball Prediction ──
    dist_to_ball = bot_player.pos.distance_to(ball.pos)
    lead_time = max(0.04, min(0.32, dist_to_ball / 750.0))
    pred_ball_pos = ball.pos + (ball.vel * lead_time)

    target = bot_player.pos
    kick = False
    kick_reach = (
        bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0
    )

    ball_in_attacking_half = (sign * (ball.pos.x - sim.center.x)) > -30.0

    # ==========================================================
    # ROLE: STRIKER (Poacher / Attacking Half Anchor)
    # ==========================================================
    if role == "STRIKER":
      if ball_in_attacking_half:
        # Actively attack ball and score
        shot_target = self._select_shot_target(
            pred_ball_pos, opp_team, opp_goal_x, sim.center.y, goal_h
        )
        shot_dir = (shot_target - pred_ball_pos).normalize()
        ideal_strike_pos = pred_ball_pos - (shot_dir * (bot_player.radius + 14.0))

        to_ball = pred_ball_pos - bot_player.pos
        dist_ball = to_ball.length()
        dir_to_ball = to_ball.normalize() if dist_ball > 0 else shot_dir

        alignment = dir_to_ball.x * shot_dir.x + dir_to_ball.y * shot_dir.y
        if alignment > 0.40:
          target = pred_ball_pos + (shot_dir * 30.0)
        else:
          cross = shot_dir.x * to_ball.y - shot_dir.y * to_ball.x
          side = 1.0 if cross >= 0 else -1.0
          perp = Vec2(-shot_dir.y * side, shot_dir.x * side)
          evasion_weight = max(0.0, 1.0 - (dist_ball / 130.0))
          target = ideal_strike_pos + (perp * 80.0 * evasion_weight)

        if bot_player.pos.distance_to(ball.pos) <= kick_reach:
          bot_to_ball = (ball.pos - bot_player.pos).normalize()
          if (bot_to_ball.x * sign) > 0.10:
            kick = True

      else:
        # Ball in defensive half: Hold attacking shape and open up passing lane
        base_anchor_x = sim.center.x + (sign * (pitch_w * 0.26))
        # Mirror ball Y slightly while staying within pitch bounds
        base_anchor_y = sim.center.y + (ball.pos.y - sim.center.y) * 0.60

        # Adjust position if opponent is directly obstructing clearance path
        test_pos = Vec2(base_anchor_x, base_anchor_y)
        if self._is_path_blocked(ball.pos, test_pos, opp_team, radius_threshold=42.0):
          offset_sign = 1.0 if base_anchor_y >= sim.center.y else -1.0
          base_anchor_y -= offset_sign * 110.0

        target = Vec2(base_anchor_x, base_anchor_y)

      # Hard constraint: Striker never crosses back into defensive half
      min_striker_x = sim.center.x + (sign * 50.0)
      if sign > 0:
        target.x = max(target.x, min_striker_x)
      else:
        target.x = min(target.x, min_striker_x)

    # ==========================================================
    # ROLES: GK & DEFENDER (Defend and Feed Striker)
    # ==========================================================
    else:
      pass_target = None
      if striker_teammate is not None:
        # Lead the striker slightly forward into the attacking channel
        pass_target = striker_teammate.pos + Vec2(sign * 25.0, 0.0)
      else:
        pass_target = Vec2(opp_goal_x, sim.center.y)

      is_pass_blocked = self._is_path_blocked(
          pred_ball_pos, pass_target, opp_team, radius_threshold=36.0
      )
      clearance_dir = (
          (pass_target - pred_ball_pos).normalize()
          if not is_pass_blocked
          else Vec2(sign, (1.0 if pred_ball_pos.y < sim.center.y else -1.0) * 0.4).normalize()
      )

      if role == "GK":
        max_gk_advance = min(220.0, pitch_w * 0.25)
        ball_in_box = (
            sign * (ball.pos.x - own_goal_x) < (pitch_w * 0.22)
            and abs(ball.pos.y - sim.center.y) < (goal_h * 1.2)
        )

        if ball_in_box:
          ideal_approach = pred_ball_pos - (clearance_dir * (bot_player.radius + 12.0))
          target = ideal_approach if bot_player.pos.distance_to(pred_ball_pos) > 40.0 else pred_ball_pos
        else:
          to_ball = ball.pos - own_goal_pos
          step_out = min(max_gk_advance, max(45.0, to_ball.length() * 0.18))
          target_x = own_goal_x + (sign * step_out)
          goal_half = (goal_h * 0.5) - 10.0
          target_y = min(
              max(sim.center.y - goal_half, ball.pos.y), sim.center.y + goal_half
          )
          target = Vec2(target_x, target_y)

        # Clamped to defensive third
        max_x_allowed = sim.center.x - (sign * (pitch_w * 0.15))
        target.x = min(target.x, max_x_allowed) if sign > 0 else max(target.x, max_x_allowed)

      elif role == "DEFENDER":
        ball_in_defensive_zone = (sign * (ball.pos.x - sim.center.x)) < 120.0

        if ball_in_defensive_zone:
          ideal_approach = pred_ball_pos - (clearance_dir * (bot_player.radius + 14.0))
          target = ideal_approach if bot_player.pos.distance_to(pred_ball_pos) > 45.0 else pred_ball_pos
        else:
          # Hold midfield anchor line
          anchor_x = sim.center.x - (sign * (pitch_w * 0.12))
          anchor_y = sim.center.y + (ball.pos.y - sim.center.y) * 0.50
          target = Vec2(anchor_x, anchor_y)

        # Defender boundary clamp
        max_def_x = sim.center.x + (sign * (pitch_w * 0.08))
        target.x = min(target.x, max_def_x) if sign > 0 else max(target.x, max_def_x)

      # Direct Clearance / Passing Trigger
      if bot_player.pos.distance_to(ball.pos) <= kick_reach:
        bot_to_ball = (ball.pos - bot_player.pos).normalize()
        kick_alignment = bot_to_ball.x * clearance_dir.x + bot_to_ball.y * clearance_dir.y
        if kick_alignment > 0.15 or (bot_to_ball.x * sign) > 0.0:
          kick = True

    # ── 3. Pitch Boundary Clamping & Motor Vector ──
    target.x = max(p.outer_left + safe_m, min(p.outer_right - safe_m, target.x))
    target.y = max(p.outer_top + safe_m, min(p.outer_bottom - safe_m, target.y))

    to_target = target - bot_player.pos
    dist_target = to_target.length()
    move_dir = to_target.normalize() if dist_target > 6.0 else Vec2(0.0, 0.0)

    return move_dir, kick