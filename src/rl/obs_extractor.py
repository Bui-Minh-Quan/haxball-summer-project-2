import numpy as np
import math
from src.engine.vector import Vec2

def extract_universal_obs(sim, player, team: str, max_teammates=4, max_opponents=5) -> np.ndarray:
    p = sim.pitch
    ball = sim.ball
    hw, hh = p.width / 2.0, p.height / 2.0
    sign = 1.0 if team == "red" else -1.0
    target_goal = Vec2(p.right if team == "red" else p.left, sim.center.y)

    # 1. SELF (8 dims)
    obs_self = [
        (player.pos.x - sim.center.x) * sign / hw,
        (player.pos.y - sim.center.y) / hh,
        np.clip(player.vel.x * sign / 1000.0, -1.0, 1.0),
        np.clip(player.vel.y / 1000.0, -1.0, 1.0),
        (target_goal.x - player.pos.x) * sign / hw,
        (target_goal.y - player.pos.y) / hh,
        player.kick_cooldown_timer / player.stats.kick_cooldown if player.stats.kick_cooldown > 0 else 0.0,
        1.0 if player.is_kicking else 0.0
    ]

    # 2. BALL (6 dims)
    obs_ball = [
        (ball.pos.x - sim.center.x) * sign / hw,
        (ball.pos.y - sim.center.y) / hh,
        np.clip(ball.vel.x * sign / 1500.0, -1.0, 1.0),
        np.clip(ball.vel.y / 1500.0, -1.0, 1.0),
        (ball.pos.x - player.pos.x) * sign / hw,
        (ball.pos.y - player.pos.y) / hh
    ]

    def _extract_others(others, max_slots):
        # Sort others by distance to the active player
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
                    1.0 # Active Mask
                ])
            else:
                slots.extend([0.0] * 6) # Empty Slot Padding
        return slots

    # 3. TEAMMATES (4 * 6 = 24 dims)
    my_team = sim.red_team if team == "red" else sim.blue_team
    teammates = [t for t in my_team if t != player]
    obs_team = _extract_others(teammates, max_teammates)

    # 4. OPPONENTS (5 * 6 = 30 dims)
    opp_team = sim.blue_team if team == "red" else sim.red_team
    obs_opp = _extract_others(opp_team, max_opponents)

    obs = np.array(obs_self + obs_ball + obs_team + obs_opp, dtype=np.float32)
    return np.clip(obs, -1.0, 1.0)