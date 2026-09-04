import math
import numpy as np
from src.engine.vector import Vec2


def extract_obs(sim, player, team: str, max_teammates: int = 4, max_opponents: int = 5) -> np.ndarray:
    p = sim.pitch
    ball = sim.ball
    hw, hh = p.width / 2.0, p.height / 2.0
    sign = 1.0 if team == "red" else -1.0

    opp_goal = Vec2(p.right if team == "red" else p.left, sim.center.y)
    own_goal = Vec2(p.left if team == "red" else p.right, sim.center.y)

    # 1. SELF STATE (10 dims)
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

    # 3. MATCH CONTEXT & TIMERS (8 dims)
    mode = sim.mode
    time_limit = getattr(mode, "time_limit", 180.0)
    time_rem = getattr(mode, "time_remaining", time_limit)
    score_limit = max(1, getattr(mode, "score_limit", 3))

    my_score = sim.score_red if team == "red" else sim.score_blue
    opp_score = sim.score_blue if team == "red" else sim.score_red

    is_kickoff = 1.0 if getattr(mode, "state", "") == "KICKOFF" else 0.0
    kickoff_team = getattr(mode, "kickoff_team", "red")
    is_our_kickoff = 1.0 if (is_kickoff and kickoff_team == team) else (-1.0 if is_kickoff else 0.0)

    kickoff_timeout = getattr(mode, "kickoff_timeout", 10.0)
    kickoff_timer = getattr(mode, "kickoff_timer", 0.0)

    obs_match = [
        np.clip(time_rem / max(1.0, time_limit), 0.0, 1.0) if time_limit > 0 else 1.0,
        np.clip(my_score / score_limit, 0.0, 1.0),
        np.clip(opp_score / score_limit, 0.0, 1.0),
        np.clip((my_score - opp_score) / score_limit, -1.0, 1.0),
        is_kickoff,
        is_our_kickoff,
        np.clip(kickoff_timer / max(1.0, kickoff_timeout), 0.0, 1.0),
        1.0 if getattr(mode, "state", "") == "GOAL_SCORED" else 0.0,
    ]

    def _extract_others(others, max_slots):
        others_sorted = sorted(others, key=lambda o: player.pos.distance_to(o.pos))
        slots = []
        for i in range(max_slots):
            if i < len(others_sorted):
                o = others_sorted[i]
                dist = player.pos.distance_to(o.pos)
                slots.extend([
                    (o.pos.x - player.pos.x) * sign / hw,
                    (o.pos.y - player.pos.y) / hh,
                    np.clip(o.vel.x * sign / 1000.0, -1.0, 1.0),
                    np.clip(o.vel.y / 1000.0, -1.0, 1.0),
                    dist / math.hypot(p.width, p.height),
                    1.0,  # Active Entity Mask
                ])
            else:
                slots.extend([0.0] * 6)  # Padded Inactive Slot
        return slots

    # 4. TEAMMATES (4 * 6 = 24 dims)
    my_team = sim.red_team if team == "red" else sim.blue_team
    teammates = [t for t in my_team if t != player]
    obs_team = _extract_others(teammates, max_teammates)

    # 5. OPPONENTS (5 * 6 = 30 dims)
    opp_team = sim.blue_team if team == "red" else sim.red_team
    obs_opp = _extract_others(opp_team, max_opponents)

    obs = np.array(obs_self + obs_ball + obs_match + obs_team + obs_opp, dtype=np.float32)
    return np.clip(obs, -1.0, 1.0)


def extract_global_state(
    sim, team: str, max_teammates: int = 2, max_opponents: int = 2
) -> np.ndarray:
    """Extracts a centralized, permutation-invariant 32-dim state for the MAPPO Critic."""
    p = sim.pitch
    ball = sim.ball
    hw, hh = p.width / 2.0, p.height / 2.0
    
    # Flip X-axis so both teams view themselves attacking from Left to Right
    sign = 1.0 if team == "red" else -1.0

    # 1. MATCH CONTEXT (4 dims)
    mode = sim.mode
    time_limit = getattr(mode, "time_limit", 180.0)
    time_rem = getattr(mode, "time_remaining", time_limit)
    score_limit = max(1, getattr(mode, "score_limit", 3))

    my_score = sim.score_red if team == "red" else sim.score_blue
    opp_score = sim.score_blue if team == "red" else sim.score_red

    obs_match = [
        np.clip(time_rem / max(1.0, time_limit), 0.0, 1.0) if time_limit > 0 else 1.0,
        np.clip(my_score / score_limit, 0.0, 1.0),
        np.clip(opp_score / score_limit, 0.0, 1.0),
        np.clip((my_score - opp_score) / score_limit, -1.0, 1.0),
    ]

    # 2. BALL STATE (4 dims)
    obs_ball = [
        (ball.pos.x - sim.center.x) * sign / hw,
        (ball.pos.y - sim.center.y) / hh,
        np.clip(ball.vel.x * sign / 1500.0, -1.0, 1.0),
        np.clip(ball.vel.y / 1500.0, -1.0, 1.0),
    ]

    def _extract_players(players, max_slots):
        # Sort by X-position in attack direction (Defenders first, Strikers last)
        players_sorted = sorted(players, key=lambda pl: (pl.pos.x - sim.center.x) * sign)
        slots = []
        for i in range(max_slots):
            if i < len(players_sorted):
                pl = players_sorted[i]
                slots.extend([
                    (pl.pos.x - sim.center.x) * sign / hw,
                    (pl.pos.y - sim.center.y) / hh,
                    np.clip(pl.vel.x * sign / 1000.0, -1.0, 1.0),
                    np.clip(pl.vel.y / 1000.0, -1.0, 1.0),
                    1.0 if pl.is_kicking else 0.0,
                    1.0,  # Active Mask
                ])
            else:
                slots.extend([0.0] * 6)  # Padded Slot
        return slots

    # 3. TEAMMATES (2 * 6 = 12 dims)
    my_team = sim.red_team if team == "red" else sim.blue_team
    obs_team = _extract_players(my_team, max_teammates)

    # 4. OPPONENTS (2 * 6 = 12 dims)
    opp_team = sim.blue_team if team == "red" else sim.red_team
    obs_opp = _extract_players(opp_team, max_opponents)

    state = np.array(obs_match + obs_ball + obs_team + obs_opp, dtype=np.float32)
    return np.clip(state, -1.0, 1.0)