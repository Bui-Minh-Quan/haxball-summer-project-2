import math
import numpy as np

from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2

# ── Frame Stacking Dimensions ──
FRAME_STACK = 3

SINGLE_EGO_DIM = 9
SINGLE_BALL_DIM = 14
SINGLE_PLAYER_DIM = 9

EGO_DIM = SINGLE_EGO_DIM * FRAME_STACK        # 27
BALL_DIM = SINGLE_BALL_DIM * FRAME_STACK      # 42
PLAYER_DIM = SINGLE_PLAYER_DIM * FRAME_STACK  # 27

MAX_TEAMMATES = 2
MAX_OPPONENTS = 5                             # Expanded from 3 to 5
TOTAL_TOKENS = 1 + 1 + MAX_TEAMMATES + MAX_OPPONENTS  # 1 Ego + 1 Ball + 2 Mates + 5 Opps = 9 Tokens
CRITIC_TOKENS = 1 + 3 + MAX_OPPONENTS                 # 1 Ball + 3 Learners + 5 Opps = 9 Tokens

def extract_entity_obs(
    sim: Simulation, ego_player: Player, team: str
) -> dict[str, np.ndarray]:
  """Tokenizes instantaneous simulation state into single-frame entity arrays."""
  p = sim.pitch
  c = sim.center
  sign = 1.0 if team == "red" else -1.0

  pitch_w = p.right - p.left
  pitch_h = p.bottom - p.top
  half_w = pitch_w * 0.5
  half_h = pitch_h * 0.5
  diag = math.hypot(pitch_w, pitch_h)
  max_speed = 1000.0

  own_goal_x = p.left if team == "red" else p.right
  opp_goal_x = p.right if team == "red" else p.left
  own_goal_pos = Vec2(own_goal_x, c.y)
  opp_goal_pos = Vec2(opp_goal_x, c.y)

  my_team = sim.red_team if team == "red" else sim.blue_team
  opp_team = sim.blue_team if team == "red" else sim.red_team
  ball = sim.ball

  # 1. EGO TOKEN (SINGLE_EGO_DIM = 9)
  ego_pos_x = ((ego_player.pos.x - c.x) / half_w) * sign
  ego_pos_y = (ego_player.pos.y - c.y) / half_h
  ego_vel_x = (ego_player.vel.x / max_speed) * sign
  ego_vel_y = ego_player.vel.y / max_speed
  dx_own = ((own_goal_x - ego_player.pos.x) / pitch_w) * sign
  dy_own = (c.y - ego_player.pos.y) / pitch_h
  dx_opp = ((opp_goal_x - ego_player.pos.x) / pitch_w) * sign
  dy_opp = (c.y - ego_player.pos.y) / pitch_h
  cooldown_raw = getattr(ego_player, "kick_cooldown_timer", 0.0)
  cooldown = min(1.0, max(0.0, cooldown_raw / 0.25))

  ego_token = np.array(
      [ego_pos_x, ego_pos_y, ego_vel_x, ego_vel_y, dx_own, dy_own, dx_opp, dy_opp, cooldown],
      dtype=np.float32,
  )

  # 2. BALL TOKEN (SINGLE_BALL_DIM = 14)
  b_rel_x = ((ball.pos.x - ego_player.pos.x) / pitch_w) * sign
  b_rel_y = (ball.pos.y - ego_player.pos.y) / pitch_h
  b_vel_x = (ball.vel.x / max_speed) * sign
  b_vel_y = ball.vel.y / max_speed
  b_rel_vx = ((ball.vel.x - ego_player.vel.x) / max_speed) * sign
  b_rel_vy = (ball.vel.y - ego_player.vel.y) / max_speed
  b_dist_ego = ego_player.pos.distance_to(ball.pos) / diag
  b_dx_opp = ((opp_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_opp = (c.y - ball.pos.y) / pitch_h
  b_dist_opp = ball.pos.distance_to(opp_goal_pos) / diag
  b_dx_own = ((own_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_own = (c.y - ball.pos.y) / pitch_h

  dx_eb = ball.pos.x - ego_player.pos.x
  dy_eb = ball.pos.y - ego_player.pos.y
  dist_eb = math.hypot(dx_eb, dy_eb)
  u_eb_x = (dx_eb / dist_eb * sign) if dist_eb > 1e-4 else 0.0
  u_eb_y = (dy_eb / dist_eb) if dist_eb > 1e-4 else 0.0

  dx_bg = opp_goal_x - ball.pos.x
  dy_bg = c.y - ball.pos.y
  dist_bg = math.hypot(dx_bg, dy_bg)
  u_bg_x = (dx_bg / dist_bg * sign) if dist_bg > 1e-4 else 1.0
  u_bg_y = (dy_bg / dist_bg) if dist_bg > 1e-4 else 0.0

  align_cos = u_eb_x * u_bg_x + u_eb_y * u_bg_y
  align_sin = u_eb_x * u_bg_y - u_eb_y * u_bg_x

  ball_token = np.array(
      [
          b_rel_x, b_rel_y, b_vel_x, b_vel_y, b_rel_vx, b_rel_vy,
          b_dist_ego, b_dx_opp, b_dy_opp, b_dist_opp, b_dx_own, b_dy_own,
          align_cos, align_sin
      ],
      dtype=np.float32,
  )

  # 3. TEAMMATES TOKENS (MAX_TEAMMATES, SINGLE_PLAYER_DIM = 9)
  teammate_tokens = np.zeros((MAX_TEAMMATES, SINGLE_PLAYER_DIM), dtype=np.float32)
  tm_mask = [True] * MAX_TEAMMATES

  teammates = [pl for pl in my_team if pl != ego_player]
  for idx, mate in enumerate(teammates[:MAX_TEAMMATES]):
    m_rel_x = ((mate.pos.x - ego_player.pos.x) / pitch_w) * sign
    m_rel_y = (mate.pos.y - ego_player.pos.y) / pitch_h
    m_vel_x = (mate.vel.x / max_speed) * sign
    m_vel_y = mate.vel.y / max_speed
    m_rel_vx = ((mate.vel.x - ego_player.vel.x) / max_speed) * sign
    m_rel_vy = (mate.vel.y - ego_player.vel.y) / max_speed
    m_dist_ego = ego_player.pos.distance_to(mate.pos) / diag
    m_dist_ball = ball.pos.distance_to(mate.pos) / diag

    teammate_tokens[idx] = [
        m_rel_x, m_rel_y, m_vel_x, m_vel_y, m_rel_vx, m_rel_vy, m_dist_ego, m_dist_ball, 1.0
    ]
    tm_mask[idx] = False

  # 4. OPPONENTS TOKENS (MAX_OPPONENTS, SINGLE_PLAYER_DIM = 9)
  opponent_tokens = np.zeros((MAX_OPPONENTS, SINGLE_PLAYER_DIM), dtype=np.float32)
  opp_mask = [True] * MAX_OPPONENTS

  for idx, opp in enumerate(opp_team[:MAX_OPPONENTS]):
    o_rel_x = ((opp.pos.x - ego_player.pos.x) / pitch_w) * sign
    o_rel_y = (opp.pos.y - ego_player.pos.y) / pitch_h
    o_vel_x = (opp.vel.x / max_speed) * sign
    o_vel_y = opp.vel.y / max_speed
    o_rel_vx = ((opp.vel.x - ego_player.vel.x) / max_speed) * sign
    o_rel_vy = (opp.vel.y - ego_player.vel.y) / max_speed
    o_dist_ego = ego_player.pos.distance_to(opp.pos) / diag
    o_dist_ball = ball.pos.distance_to(opp.pos) / diag

    opponent_tokens[idx] = [
        o_rel_x, o_rel_y, o_vel_x, o_vel_y, o_rel_vx, o_rel_vy, o_dist_ego, o_dist_ball, 1.0
    ]
    opp_mask[idx] = False

  # 5. KEY PADDING MASK (7,)
  key_padding_mask = np.array([False, False] + tm_mask + opp_mask, dtype=bool)

  return {
      "ego": ego_token,
      "ball": ball_token,
      "teammates": teammate_tokens,
      "opponents": opponent_tokens,
      "key_padding_mask": key_padding_mask,
  }

def extract_global_critic_entities(
    sim: Simulation, learner_team: str
) -> dict[str, np.ndarray]:
  """Provides pitch-centric single-frame tokenized entities for Centralized Critic."""
  p = sim.pitch
  c = sim.center
  sign = 1.0 if learner_team == "red" else -1.0

  pitch_w = p.right - p.left
  pitch_h = p.bottom - p.top
  half_w = pitch_w * 0.5
  half_h = pitch_h * 0.5
  diag = math.hypot(pitch_w, pitch_h)
  max_speed = 1000.0

  own_goal_x = p.left if learner_team == "red" else p.right
  opp_goal_x = p.right if learner_team == "red" else p.left
  own_goal_pos = Vec2(own_goal_x, c.y)
  opp_goal_pos = Vec2(opp_goal_x, c.y)

  my_team = sim.red_team if learner_team == "red" else sim.blue_team
  opp_team = sim.blue_team if learner_team == "red" else sim.red_team
  ball = sim.ball

  # Ball Token (SINGLE_BALL_DIM = 14)
  b_pos_x = ((ball.pos.x - c.x) / half_w) * sign
  b_pos_y = (ball.pos.y - c.y) / half_h
  b_vel_x = (ball.vel.x / max_speed) * sign
  b_vel_y = ball.vel.y / max_speed
  b_speed = math.hypot(ball.vel.x, ball.vel.y) / max_speed
  b_dist_center = ball.pos.distance_to(c) / diag
  b_dx_opp = ((opp_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_opp = (c.y - ball.pos.y) / pitch_h
  b_dist_opp = ball.pos.distance_to(opp_goal_pos) / diag
  b_dx_own = ((own_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_own = (c.y - ball.pos.y) / pitch_h
  b_dist_own = ball.pos.distance_to(own_goal_pos) / diag
  to_opp_x = (opp_goal_x - ball.pos.x) * sign
  to_opp_y = c.y - ball.pos.y
  b_vel_to_opp = (b_vel_x * to_opp_x + b_vel_y * to_opp_y) / (diag * max_speed)
  b_in_own_half = 1.0 if ((ball.pos.x - c.x) * sign < 0) else -1.0

  ball_token = np.array(
      [
          b_pos_x, b_pos_y, b_vel_x, b_vel_y, b_speed, b_dist_center,
          b_dx_opp, b_dy_opp, b_dist_opp, b_dx_own, b_dy_own, b_dist_own,
          b_vel_to_opp, b_in_own_half
      ],
      dtype=np.float32,
  )

  # Player Tokens (SINGLE_PLAYER_DIM = 9)
  def _encode_player(pl: Player | None) -> np.ndarray:
    if pl is None:
      return np.zeros(SINGLE_PLAYER_DIM, dtype=np.float32)
    px = ((pl.pos.x - c.x) / half_w) * sign
    py = (pl.pos.y - c.y) / half_h
    vx = (pl.vel.x / max_speed) * sign
    vy = pl.vel.y / max_speed
    speed = math.hypot(pl.vel.x, pl.vel.y) / max_speed
    d_center = pl.pos.distance_to(c) / diag
    d_ball = pl.pos.distance_to(ball.pos) / diag
    d_opp_goal = pl.pos.distance_to(opp_goal_pos) / diag
    return np.array([px, py, vx, vy, speed, d_center, d_ball, d_opp_goal, 1.0], dtype=np.float32)

  learners_tokens = np.zeros((3, SINGLE_PLAYER_DIM), dtype=np.float32)
  learners_mask = [True] * 3
  for idx, pl in enumerate(my_team[:3]):
    learners_tokens[idx] = _encode_player(pl)
    learners_mask[idx] = False

  opponents_tokens = np.zeros((MAX_OPPONENTS, SINGLE_PLAYER_DIM), dtype=np.float32)
  opponents_mask = [True] * MAX_OPPONENTS
  for idx, opp in enumerate(opp_team[:MAX_OPPONENTS]):
    opponents_tokens[idx] = _encode_player(opp)
    opponents_mask[idx] = False

  key_padding_mask = np.array([False] + learners_mask + opponents_mask, dtype=bool)

  return {
      "ball": ball_token,
      "learners": learners_tokens,
      "opponents": opponents_tokens,
      "key_padding_mask": key_padding_mask,
  }