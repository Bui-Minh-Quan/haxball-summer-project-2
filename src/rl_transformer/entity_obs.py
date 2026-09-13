import math
import numpy as np

from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2

# Entity Token Shapes
EGO_DIM = 9
BALL_DIM = 10
PLAYER_DIM = 7
MAX_TEAMMATES = 2
MAX_OPPONENTS = 3
TOTAL_TOKENS = 1 + 1 + MAX_TEAMMATES + MAX_OPPONENTS  # 7 Tokens


def extract_entity_obs(
    sim: Simulation, ego_player: Player, team: str
) -> dict[str, np.ndarray]:
  """Tokenizes the simulation state into structured entity arrays for the Entity-Transformer.

  Returns:
      ego: (9,)
      ball: (10,)
      teammates: (2, 7)
      opponents: (3, 7)
      key_padding_mask: (7,) boolean mask (True = ignored/padded, False = active)
  """
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

  # =========================================================================
  # 1. EGO TOKEN (9,)
  # =========================================================================
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
      [
          ego_pos_x,
          ego_pos_y,
          ego_vel_x,
          ego_vel_y,
          dx_own,
          dy_own,
          dx_opp,
          dy_opp,
          cooldown,
      ],
      dtype=np.float32,
  )

  # =========================================================================
  # 2. BALL TOKEN (10,)
  # =========================================================================
  b_rel_x = ((ball.pos.x - ego_player.pos.x) / pitch_w) * sign
  b_rel_y = (ball.pos.y - ego_player.pos.y) / pitch_h

  b_vel_x = (ball.vel.x / max_speed) * sign
  b_vel_y = ball.vel.y / max_speed

  b_dist_ego = ego_player.pos.distance_to(ball.pos) / diag

  b_dx_opp = ((opp_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_opp = (c.y - ball.pos.y) / pitch_h
  b_dist_opp = ball.pos.distance_to(opp_goal_pos) / diag

  b_dx_own = ((own_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_own = (c.y - ball.pos.y) / pitch_h

  ball_token = np.array(
      [
          b_rel_x,
          b_rel_y,
          b_vel_x,
          b_vel_y,
          b_dist_ego,
          b_dx_opp,
          b_dy_opp,
          b_dist_opp,
          b_dx_own,
          b_dy_own,
      ],
      dtype=np.float32,
  )

  # =========================================================================
  # 3. TEAMMATES TOKENS (2, 7)
  # =========================================================================
  teammate_tokens = np.zeros((MAX_TEAMMATES, PLAYER_DIM), dtype=np.float32)
  tm_mask = [True] * MAX_TEAMMATES  # True = padded

  teammates = [pl for pl in my_team if pl != ego_player]
  for idx, mate in enumerate(teammates[:MAX_TEAMMATES]):
    m_rel_x = ((mate.pos.x - ego_player.pos.x) / pitch_w) * sign
    m_rel_y = (mate.pos.y - ego_player.pos.y) / pitch_h

    m_vel_x = (mate.vel.x / max_speed) * sign
    m_vel_y = mate.vel.y / max_speed

    m_dist_ego = ego_player.pos.distance_to(mate.pos) / diag
    m_dist_ball = ball.pos.distance_to(mate.pos) / diag

    teammate_tokens[idx] = [
        m_rel_x,
        m_rel_y,
        m_vel_x,
        m_vel_y,
        m_dist_ego,
        m_dist_ball,
        1.0,  # Active presence indicator
    ]
    tm_mask[idx] = False

  # =========================================================================
  # 4. OPPONENTS TOKENS (3, 7)
  # =========================================================================
  opponent_tokens = np.zeros((MAX_OPPONENTS, PLAYER_DIM), dtype=np.float32)
  opp_mask = [True] * MAX_OPPONENTS  # True = padded

  for idx, opp in enumerate(opp_team[:MAX_OPPONENTS]):
    o_rel_x = ((opp.pos.x - ego_player.pos.x) / pitch_w) * sign
    o_rel_y = (opp.pos.y - ego_player.pos.y) / pitch_h

    o_vel_x = (opp.vel.x / max_speed) * sign
    o_vel_y = opp.vel.y / max_speed

    o_dist_ego = ego_player.pos.distance_to(opp.pos) / diag
    o_dist_ball = ball.pos.distance_to(opp.pos) / diag

    opponent_tokens[idx] = [
        o_rel_x,
        o_rel_y,
        o_vel_x,
        o_vel_y,
        o_dist_ego,
        o_dist_ball,
        1.0,  # Active presence indicator
    ]
    opp_mask[idx] = False

  # =========================================================================
  # 5. KEY PADDING MASK (7,) - Follows PyTorch nn.MultiheadAttention convention
  # =========================================================================
  # False = Attend, True = Ignore (Padded out)
  key_padding_mask = np.array(
      [False, False] + tm_mask + opp_mask,
      dtype=bool,
  )

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
  """Provides pitch-centric tokenized entities for a Centralized Transformer Critic.

  Positions and velocities are expressed relative to pitch center.
  """
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

  # Ball from pitch center (matches BALL_DIM=10)
  b_pos_x = ((ball.pos.x - c.x) / half_w) * sign
  b_pos_y = (ball.pos.y - c.y) / half_h
  b_vel_x = (ball.vel.x / max_speed) * sign
  b_vel_y = ball.vel.y / max_speed
  b_dist_center = ball.pos.distance_to(c) / diag
  b_dx_opp = ((opp_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_opp = (c.y - ball.pos.y) / pitch_h
  b_dist_opp = ball.pos.distance_to(opp_goal_pos) / diag
  b_dx_own = ((own_goal_x - ball.pos.x) / pitch_w) * sign
  b_dy_own = (c.y - ball.pos.y) / pitch_h

  ball_token = np.array(
      [
          b_pos_x,
          b_pos_y,
          b_vel_x,
          b_vel_y,
          b_dist_center,
          b_dx_opp,
          b_dy_opp,
          b_dist_opp,
          b_dx_own,
          b_dy_own,
      ],
      dtype=np.float32,
  )

  # All players mapped as pitch-centric tokens (matches PLAYER_DIM=7)
  def _encode_player(pl: Player | None) -> np.ndarray:
    if pl is None:
      return np.zeros(PLAYER_DIM, dtype=np.float32)
    px = ((pl.pos.x - c.x) / half_w) * sign
    py = (pl.pos.y - c.y) / half_h
    vx = (pl.vel.x / max_speed) * sign
    vy = pl.vel.y / max_speed
    d_center = pl.pos.distance_to(c) / diag
    d_ball = pl.pos.distance_to(ball.pos) / diag
    return np.array([px, py, vx, vy, d_center, d_ball, 1.0], dtype=np.float32)

  learners_tokens = np.zeros((3, PLAYER_DIM), dtype=np.float32)
  learners_mask = [True] * 3
  for idx, pl in enumerate(my_team[:3]):
    learners_tokens[idx] = _encode_player(pl)
    learners_mask[idx] = False

  opponents_tokens = np.zeros((3, PLAYER_DIM), dtype=np.float32)
  opponents_mask = [True] * 3
  for idx, opp in enumerate(opp_team[:3]):
    opponents_tokens[idx] = _encode_player(opp)
    opponents_mask[idx] = False

  key_padding_mask = np.array(
      [False] + learners_mask + opponents_mask,
      dtype=bool,
  )

  return {
      "ball": ball_token,
      "learners": learners_tokens,
      "opponents": opponents_tokens,
      "key_padding_mask": key_padding_mask,
  }