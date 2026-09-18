import math
import random
from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class TeamHeuristicCoordinator:
  """Tactical Heuristic Coordinator with Momentum Braking and Scalable Formations.
  - Supports dynamic archetypes: 'attacking', 'defending', 'balanced'.
  - Scales roles across team sizes 1 to 5.
  - Implements PD arrival steering for goalkeepers to prevent goal-line overshoot.
  """

  def __init__(self, team: str = "blue", strategy: str = "random"):
    self.team = team.lower()
    self.attack_sign = 1.0 if self.team == "red" else -1.0
    self.strategy_mode = strategy
    self.current_strategy = "balanced"
    self._bot_state: dict[int, str] = {}
    self.reset_strategy()

  def reset_strategy(self, seed: int | None = None):
    """Samples or fixes the match tactical stance."""
    if seed is not None:
      random.seed(seed)
    if self.strategy_mode in ("attacking", "defending", "balanced"):
      self.current_strategy = self.strategy_mode
    else:
      # 40% Balanced, 30% Park-the-bus, 30% Ultra-aggressive
      roll = random.random()
      if roll < 0.40:
        self.current_strategy = "balanced"
      elif roll < 0.70:
        self.current_strategy = "defending"
      else:
        self.current_strategy = "attacking"

  def _determine_role(self, bot_player: Player, my_team: list[Player]) -> str:
    n = len(my_team)
    try:
      idx = my_team.index(bot_player)
    except ValueError:
      return "STRIKER"

    # Team size 1: Always hunts the ball
    if n == 1:
      return "STRIKER"

    strat = self.current_strategy

    # Team size 2
    if n == 2:
      if strat == "attacking":
        return "STRIKER"  # Dual press
      elif strat == "defending":
        return "GK" if idx == 0 else "DEFENDER"  # Double lock
      return "GK" if idx == 0 else "STRIKER"  # Balanced

    # Team size 3
    if n == 3:
      if strat == "attacking":
        return "DEFENDER" if idx == 0 else "STRIKER"
      elif strat == "defending":
        return "GK" if idx == 0 else "DEFENDER"
      # Balanced: 1 GK, 1 DEF, 1 STRIKER
      return "GK" if idx == 0 else ("DEFENDER" if idx == 1 else "STRIKER")

    # Team size 4
    if n == 4:
      if strat == "attacking":
        return "GK" if idx == 0 else ("MID" if idx == 1 else "STRIKER")
      elif strat == "defending":
        return "GK" if idx == 0 else ("DEFENDER" if idx <= 2 else "MID")
      # Balanced: 1 GK, 1 DEF, 1 MID, 1 STRIKER
      roles = ["GK", "DEFENDER", "MID", "STRIKER"]
      return roles[idx]

    # Team size 5
    if n >= 5:
      if strat == "attacking":
        return "GK" if idx == 0 else ("DEFENDER" if idx == 1 else ("MID" if idx == 2 else "STRIKER"))
      elif strat == "defending":
        return "GK" if idx == 0 else ("DEFENDER" if idx <= 3 else "MID")
      # Balanced: 1 GK, 2 DEF, 1 MID, 1 STRIKER
      roles = ["GK", "DEFENDER", "DEFENDER", "MID", "STRIKER"]
      return roles[min(idx, len(roles) - 1)]

    return "STRIKER"

  # ── 1. PD Arrival Steering & Momentum Deceleration ──
  def _steer_with_braking(
      self, bot: Player, target: Vec2, slow_radius: float = 120.0, max_speed: float = 400.0
  ) -> Vec2:
    """Calculates steering throttle with active counter-thrust to prevent overshoot."""
    to_target = target - bot.pos
    dist = to_target.length()

    if dist < 4.0:
      # Target reached: active counter-thrust against current drift
      if bot.vel.length() > 15.0:
        norm = bot.vel.normalize()
        return Vec2(-norm.x, -norm.y)
      return Vec2(0.0, 0.0)

    # Calculate desired velocity magnitude (Ramp down inside slow_radius)
    if dist < slow_radius:
      desired_speed = max_speed * (dist / slow_radius)
    else:
      desired_speed = max_speed

    desired_vel = to_target.normalize() * desired_speed
    steering = desired_vel - bot.vel

    return steering.normalize() if steering.length() > 0.05 else Vec2(0.0, 0.0)

  # ── 2. Striker Controller (Decisive Lead Tracking) ──
  def _get_striker_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
    ball = sim.ball
    p = sim.pitch
    bot_id = id(bot_player)
    current_state = self._bot_state.get(bot_id, "APPROACH")

    dist_to_ball = bot_player.pos.distance_to(ball.pos)
    lead_t = max(0.04, min(0.25, dist_to_ball / 900.0))
    pred_ball_x = ball.pos.x + (ball.vel.x * lead_t)
    pred_ball_y = ball.pos.y + (ball.vel.y * lead_t)

    dx = (pred_ball_x - bot_player.pos.x) * self.attack_sign
    dy = pred_ball_y - bot_player.pos.y

    if current_state == "APPROACH":
      if dx > 25.0 and abs(dy) < 55.0:
        current_state = "STRIKE"
    elif current_state == "STRIKE":
      if dx < -30.0:
        current_state = "APPROACH"

    self._bot_state[bot_id] = current_state
    behind_x = pred_ball_x - (self.attack_sign * 50.0)

    if current_state == "STRIKE":
      target_x = pred_ball_x + (self.attack_sign * 50.0)
      target_y = pred_ball_y
    else:
      if dx < 15.0:
        infield_y_sign = -1.0 if ball.pos.y > sim.center.y else 1.0
        target_x = behind_x
        target_y = ball.pos.y + (infield_y_sign * 75.0)
      else:
        target_x = behind_x
        target_y = pred_ball_y

    target_x = max(p.outer_left + 30.0, min(p.outer_right - 30.0, target_x))
    target_y = max(p.outer_top + 30.0, min(p.outer_bottom - 30.0, target_y))

    move_dir = self._steer_with_braking(bot_player, Vec2(target_x, target_y), slow_radius=80.0)

    kick_reach = bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0
    kick = dist_to_ball <= kick_reach and ((ball.pos.x - bot_player.pos.x) * self.attack_sign) > -2.0

    return move_dir, kick

  # ── 3. Goalkeeper Controller (No-Overshoot Goal Protection) ──
  def _get_gk_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
    ball = sim.ball
    p = sim.pitch
    own_goal_x = p.left if self.team == "red" else p.right
    goal_half = getattr(p, "goal_height", 220.0) * 0.5 - 20.0

    dist_ball_to_goal = abs(ball.pos.x - own_goal_x)
    ball_in_danger = dist_ball_to_goal < 220.0 and abs(ball.pos.y - sim.center.y) < (goal_half + 40.0)

    # Clear loose balls inside small penalty box
    if ball_in_danger and dist_ball_to_goal > 45.0:
      return self._get_striker_action(bot_player, sim)

    # Goal line clamp: Safe margin outside net (bot center stays strictly in-pitch)
    safe_x = own_goal_x + (self.attack_sign * (bot_player.radius + 15.0))

    # Predictive shot intercept on goal line
    if abs(ball.vel.x) > 30.0:
      t_to_goal = (safe_x - ball.pos.x) / (ball.vel.x + 1e-6)
      if 0.0 < t_to_goal < 1.0:
        pred_y = ball.pos.y + (ball.vel.y * t_to_goal)
        target_y = min(max(sim.center.y - goal_half, pred_y), sim.center.y + goal_half)
      else:
        target_y = min(max(sim.center.y - goal_half, ball.pos.y), sim.center.y + goal_half)
    else:
      target_y = min(max(sim.center.y - goal_half, ball.pos.y), sim.center.y + goal_half)

    target = Vec2(safe_x, target_y)
    move_dir = self._steer_with_braking(bot_player, target, slow_radius=90.0, max_speed=350.0)

    # Panic clearance
    to_ball = ball.pos - bot_player.pos
    kick_reach = bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0
    kick = to_ball.length() <= kick_reach and (to_ball.x * self.attack_sign) > -5.0

    return move_dir, kick

  # ── 4. Defender Controller (Bus-Parking Blockers) ──
  def _get_defender_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
    ball = sim.ball
    c = sim.center

    # Press if ball breaches deep defensive half
    ball_in_deep_def = (ball.pos.x - c.x) * self.attack_sign < -80.0
    if ball_in_deep_def:
      return self._get_striker_action(bot_player, sim)

    # Stand firm at defensive anchor box
    anchor_x = c.x - (self.attack_sign * 220.0)
    anchor_y = c.y + (ball.pos.y - c.y) * 0.65

    move_dir = self._steer_with_braking(bot_player, Vec2(anchor_x, anchor_y), slow_radius=110.0)

    to_ball = ball.pos - bot_player.pos
    kick_reach = bot_player.radius + ball.radius + bot_player.stats.kick_margin + 5.0
    kick = to_ball.length() <= kick_reach and (to_ball.x * self.attack_sign) > -3.0

    return move_dir, kick

  # ── 5. Midfielder Controller (Transition Pivot) ──
  def _get_mid_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
    ball = sim.ball
    c = sim.center

    ball_in_mid = abs(ball.pos.x - c.x) < 180.0
    if ball_in_mid:
      return self._get_striker_action(bot_player, sim)

    target_x = c.x + (self.attack_sign * 60.0)
    target_y = c.y + (ball.pos.y - c.y) * 0.45

    move_dir = self._steer_with_braking(bot_player, Vec2(target_x, target_y), slow_radius=120.0)
    return move_dir, False

  def get_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
    my_team = sim.red_team if self.team == "red" else sim.blue_team
    role = self._determine_role(bot_player, my_team)

    if role == "STRIKER":
      return self._get_striker_action(bot_player, sim)
    elif role == "GK":
      return self._get_gk_action(bot_player, sim)
    elif role == "DEFENDER":
      return self._get_defender_action(bot_player, sim)
    else:
      return self._get_mid_action(bot_player, sim)