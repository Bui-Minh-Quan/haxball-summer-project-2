import math
from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class TeamHeuristicCoordinator:
    """Decisive, non-orbiting heuristic coordinator.
    - Uses predictive ball interception instead of trailing behind.
    - Hysteresis state machine (APPROACH vs. STRIKE) eliminates tail-chasing.
    - Infield-biased flanking prevents getting stuck against walls.
    """

    def __init__(self, team: str = "blue"):
        self.team = team.lower()
        # Red attacks +X (right); Blue attacks -X (left)
        self.attack_sign = 1.0 if self.team == "red" else -1.0
        # Persistent state per bot instance: 'APPROACH' or 'STRIKE'
        self._bot_state: dict[int, str] = {}

    def _determine_role(self, bot_player: Player, my_team: list[Player]) -> str:
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

    def _get_striker_action(
        self, bot_player: Player, sim: Simulation
    ) -> tuple[Vec2, bool]:
        ball = sim.ball
        p = sim.pitch
        bot_id = id(bot_player)
        current_state = self._bot_state.get(bot_id, "APPROACH")

        # 1. Predictive Interception (Lead the ball so we don't chase its tail)
        dist_to_ball = bot_player.pos.distance_to(ball.pos)
        lead_t = max(0.04, min(0.28, dist_to_ball / 850.0))
        pred_ball_x = ball.pos.x + (ball.vel.x * lead_t)
        pred_ball_y = ball.pos.y + (ball.vel.y * lead_t)

        # Longitudinal and lateral offsets relative to attack direction
        # dx > 0 means the ball is ahead of the bot in the attack direction (bot is behind ball)
        dx = (pred_ball_x - bot_player.pos.x) * self.attack_sign
        dy = pred_ball_y - bot_player.pos.y

        # 2. Hysteresis State Machine (Eliminates oscillation)
        if current_state == "APPROACH":
            # Only enter STRIKE when solidly behind the ball and lined up in Y
            if dx > 25.0 and abs(dy) < 50.0:
                current_state = "STRIKE"
        elif current_state == "STRIKE":
            # Only drop out of STRIKE if we have completely overshot past the ball
            if dx < -30.0:
                current_state = "APPROACH"

        self._bot_state[bot_id] = current_state

        # 3. Target Calculation Based on State
        behind_x = pred_ball_x - (self.attack_sign * 55.0)

        if current_state == "STRIKE":
            # Drive straight through the ball into the opponent goal
            target_x = pred_ball_x + (self.attack_sign * 40.0)
            target_y = pred_ball_y
        else:
            # APPROACH: Loop behind the ball
            if dx < 15.0:
                # Bot is in front: Flank toward the pitch center (never into a wall)
                infield_y_sign = -1.0 if ball.pos.y > sim.center.y else 1.0
                target_x = behind_x
                target_y = ball.pos.y + (infield_y_sign * 75.0)
            else:
                # Bot is already behind in X, just slide into alignment in Y
                target_x = behind_x
                target_y = pred_ball_y

        # 4. Field Clamping & Direct Throttle
        target_x = max(p.outer_left + 25.0, min(p.outer_right - 25.0, target_x))
        target_y = max(p.outer_top + 25.0, min(p.outer_bottom - 25.0, target_y))

        move_vec = Vec2(target_x, target_y) - bot_player.pos
        move_dir = move_vec.normalize() if move_vec.length() > 3.0 else Vec2(0.0, 0.0)

        # 5. Kick Trigger
        kick = False
        kick_reach = bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0
        if dist_to_ball <= kick_reach:
            # Kick only if impulse moves ball toward opponent net (prevents own-goals)
            if ((ball.pos.x - bot_player.pos.x) * self.attack_sign) > -3.0:
                kick = True

        return move_dir, kick

    def get_action(
        self, bot_player: Player, sim: Simulation
    ) -> tuple[Vec2, bool]:
        my_team = sim.red_team if self.team == "red" else sim.blue_team
        role = self._determine_role(bot_player, my_team)

        ball = sim.ball
        p = sim.pitch
        own_goal_x = p.left if self.team == "red" else p.right

        # ── 1. STRIKER / 1v1 (Pure Ball Hunter) ──
        if role == "STRIKER":
            return self._get_striker_action(bot_player, sim)

        # ── 2. GOALKEEPER (2v2 & 3v3) ──
        elif role == "GK":
            dist_ball_to_goal = abs(ball.pos.x - own_goal_x)
            # If ball enters the danger box, charge out and clear like a striker
            if dist_ball_to_goal < 230.0 and abs(ball.pos.y - sim.center.y) < 180.0:
                return self._get_striker_action(bot_player, sim)

            # Otherwise, patrol the goal line
            goal_half = getattr(p, "goal_height", 220.0) * 0.5 - 15.0
            target_x = own_goal_x + self.attack_sign * 55.0
            target_y = min(max(sim.center.y - goal_half, ball.pos.y), sim.center.y + goal_half)

            move_vec = Vec2(target_x, target_y) - bot_player.pos
            move_dir = move_vec.normalize() if move_vec.length() > 3.0 else Vec2(0.0, 0.0)

            to_ball = ball.pos - bot_player.pos
            kick_reach = bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0
            kick = to_ball.length() <= kick_reach and (to_ball.x * self.attack_sign) > -2.0
            return move_dir, kick

        # ── 3. DEFENDER (3v3 Midfield Anchor) ──
        else:
            # Challenge any ball that crosses into the defensive half
            ball_in_our_half = (ball.pos.x - sim.center.x) * self.attack_sign < 60.0
            if ball_in_our_half:
                return self._get_striker_action(bot_player, sim)

            # Hold midfield support anchor
            target_x = sim.center.x - (self.attack_sign * 110.0)
            target_y = sim.center.y + (ball.pos.y - sim.center.y) * 0.55

            move_vec = Vec2(target_x, target_y) - bot_player.pos
            move_dir = move_vec.normalize() if move_vec.length() > 3.0 else Vec2(0.0, 0.0)
            return move_dir, False