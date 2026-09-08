import math
import numpy as np
from src.engine.vector import Vec2

# Global configuration constants
MAX_TEAMMATES = 2  # Allows up to 3v3 (Self + 2 Teammates)
MAX_OPPONENTS = 4  # Allows 3v3 plus headroom for uneven drills
ACTOR_OBS_DIM = 64
CRITIC_STATE_DIM = 58

def extract_actor_obs(sim, player, team: str) -> np.ndarray:
    """Extracts a permutation-stable, fixed-slot ego observation for an Actor (64 dims)."""
    p = sim.pitch
    ball = sim.ball
    hw, hh = p.width / 2.0, p.height / 2.0
    pitch_diag = math.hypot(p.width, p.height)
    sign = 1.0 if team == "red" else -1.0

    opp_goal = Vec2(p.right if team == "red" else p.left, sim.center.y)
    own_goal = Vec2(p.left if team == "red" else p.right, sim.center.y)

    # 1. SELF EGO-STATE (10 dims)
    obs_self = [
        (player.pos.x - sim.center.x) * sign / hw,
        (player.pos.y - sim.center.y) / hh,
        np.clip(player.vel.x * sign / 1000.0, -1.0, 1.0),
        np.clip(player.vel.y / 1000.0, -1.0, 1.0),
        (opp_goal.x - player.pos.x) * sign / hw,
        (opp_goal.y - player.pos.y) / hh,
        (own_goal.x - player.pos.x) * sign / hw,
        (own_goal.y - player.pos.y) / hh,
        player.kick_cooldown_timer / player.stats.kick_cooldown if player.stats.kick_cooldown > 0 else 0.0,
        1.0 if player.is_kicking else 0.0,
    ]

    # 2. BALL STATE (8 dims)
    obs_ball = [
        (ball.pos.x - sim.center.x) * sign / hw,
        (ball.pos.y - sim.center.y) / hh,
        np.clip(ball.vel.x * sign / 1500.0, -1.0, 1.0),
        np.clip(ball.vel.y / 1500.0, -1.0, 1.0),
        (ball.pos.x - player.pos.x) * sign / hw,
        (ball.pos.y - player.pos.y) / hh,
        (opp_goal.x - ball.pos.x) * sign / hw,
        (opp_goal.y - ball.pos.y) / hh,
    ]

    # 3. MATCH SCENARIO CONTEXT (4 dims)
    mode = sim.mode
    time_limit = getattr(mode, "time_limit", 60.0)
    time_rem = getattr(mode, "time_remaining", time_limit)
    score_limit = max(1, getattr(mode, "score_limit", 5))

    my_score = sim.score_red if team == "red" else sim.score_blue
    opp_score = sim.score_blue if team == "red" else sim.score_red

    obs_context = [
        np.clip(time_rem / max(1.0, time_limit), 0.0, 1.0),
        np.clip(my_score / float(score_limit), 0.0, 1.0),
        np.clip(opp_score / float(score_limit), 0.0, 1.0),
        np.clip((my_score - opp_score) / float(score_limit), -1.0, 1.0),
    ]

    # 4. FIXED-SLOT TEAMMATES (2 slots * 7 dims = 14 dims)
    my_squad = sim.red_team if team == "red" else sim.blue_team
    # Exclude self while retaining deterministic list order
    teammates = [pl for pl in my_squad if pl != player]

    obs_teammates = []
    for slot_idx in range(MAX_TEAMMATES):
        if slot_idx < len(teammates):
            tm = teammates[slot_idx]
            dist = player.pos.distance_to(tm.pos)
            obs_teammates.extend([
                (tm.pos.x - player.pos.x) * sign / hw,
                (tm.pos.y - player.pos.y) / hh,
                np.clip(tm.vel.x * sign / 1000.0, -1.0, 1.0),
                np.clip(tm.vel.y / 1000.0, -1.0, 1.0),
                dist / pitch_diag,
                1.0 if tm.is_kicking else 0.0,
                1.0,  # Active Mask
            ])
        else:
            obs_teammates.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 5. FIXED-SLOT OPPONENTS (4 slots * 7 dims = 28 dims)
    opp_squad = sim.blue_team if team == "red" else sim.red_team

    obs_opponents = []
    for slot_idx in range(MAX_OPPONENTS):
        if slot_idx < len(opp_squad):
            opp = opp_squad[slot_idx]
            dist = player.pos.distance_to(opp.pos)
            obs_opponents.extend([
                (opp.pos.x - player.pos.x) * sign / hw,
                (opp.pos.y - player.pos.y) / hh,
                np.clip(opp.vel.x * sign / 1000.0, -1.0, 1.0),
                np.clip(opp.vel.y / 1000.0, -1.0, 1.0),
                dist / pitch_diag,
                1.0 if opp.is_kicking else 0.0,
                1.0,  # Active Mask
            ])
        else:
            obs_opponents.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    total_obs = obs_self + obs_ball + obs_context + obs_teammates + obs_opponents
    return np.clip(np.array(total_obs, dtype=np.float32), -1.0, 1.0)


def extract_global_state(sim, team: str) -> np.ndarray:
    """Extracts a centralized global state for the Critic (58 dims)."""
    p = sim.pitch
    ball = sim.ball
    hw, hh = p.width / 2.0, p.height / 2.0
    pitch_diag = math.hypot(p.width, p.height)
    sign = 1.0 if team == "red" else -1.0

    opp_goal = Vec2(p.right if team == "red" else p.left, sim.center.y)
    own_goal = Vec2(p.left if team == "red" else p.right, sim.center.y)

    # 1. MATCH SCENARIO CONTEXT (4 dims)
    mode = sim.mode
    time_limit = getattr(mode, "time_limit", 60.0)
    time_rem = getattr(mode, "time_remaining", time_limit)
    score_limit = max(1, getattr(mode, "score_limit", 5))

    my_score = sim.score_red if team == "red" else sim.score_blue
    opp_score = sim.score_blue if team == "red" else sim.score_red

    state_context = [
        np.clip(time_rem / max(1.0, time_limit), 0.0, 1.0),
        np.clip(my_score / float(score_limit), 0.0, 1.0),
        np.clip(opp_score / float(score_limit), 0.0, 1.0),
        np.clip((my_score - opp_score) / float(score_limit), -1.0, 1.0),
    ]

    # 2. BALL GLOBAL STATE (6 dims)
    state_ball = [
        (ball.pos.x - sim.center.x) * sign / hw,
        (ball.pos.y - sim.center.y) / hh,
        np.clip(ball.vel.x * sign / 1500.0, -1.0, 1.0),
        np.clip(ball.vel.y / 1500.0, -1.0, 1.0),
        ball.pos.distance_to(opp_goal) / pitch_diag,
        ball.pos.distance_to(own_goal) / pitch_diag,
    ]

    # Helper for deterministic fixed-slot player extraction (8 dims per slot)
    def _extract_team_slots(players, max_slots=3):
        slots = []
        for i in range(max_slots):
            if i < len(players):
                pl = players[i]
                slots.extend([
                    (pl.pos.x - sim.center.x) * sign / hw,
                    (pl.pos.y - sim.center.y) / hh,
                    np.clip(pl.vel.x * sign / 1000.0, -1.0, 1.0),
                    np.clip(pl.vel.y / 1000.0, -1.0, 1.0),
                    pl.kick_cooldown_timer / pl.stats.kick_cooldown if pl.stats.kick_cooldown > 0 else 0.0,
                    1.0 if pl.is_kicking else 0.0,
                    (i + 1.0) / float(max_slots),  # Slot index identifier
                    1.0,  # Active Mask
                ])
            else:
                slots.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return slots

    my_squad = sim.red_team if team == "red" else sim.blue_team
    opp_squad = sim.blue_team if team == "red" else sim.red_team

    # 3. LEARNER SQUAD (3 slots * 8 dims = 24 dims)
    state_my_team = _extract_team_slots(my_squad, max_slots=3)

    # 4. OPPONENT SQUAD (3 slots * 8 dims = 24 dims)
    state_opp_team = _extract_team_slots(opp_squad, max_slots=3)

    total_state = state_context + state_ball + state_my_team + state_opp_team
    return np.clip(np.array(total_state, dtype=np.float32), -1.0, 1.0)