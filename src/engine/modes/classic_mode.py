from typing import Any
import math
from src.engine.modes.base_mode import GameMode
from config.match_config import MatchConfig
from src.engine.vector import Vec2


class ClassicMatchMode(GameMode):

    def __init__(self, time_limit: float = 180.0, score_limit: int = 3, kickoff_timeout: float = 10.0):
        self.time_limit = time_limit
        self.score_limit = score_limit
        self.kickoff_timeout = kickoff_timeout
        self.time_remaining = time_limit
        self.state = "KICKOFF"  # "KICKOFF", "PLAYING", "GOAL_SCORED"
        self.kickoff_team = "red"
        self.state_timer = 0.0
        self.kickoff_timer = 0.0

    def init_mode(self, sim: Any):
        # Sync parameters from match_config if provided
        if hasattr(sim, "match_config") and sim.match_config:
            self.time_limit = sim.match_config.time_limit
            self.score_limit = sim.match_config.score_limit
            self.kickoff_timeout = sim.match_config.kickoff_timeout

        sim.score_red = 0
        sim.score_blue = 0
        self.time_remaining = self.time_limit
        self.state = "KICKOFF"
        self.kickoff_team = "red"
        self.kickoff_timer = 0.0
        self.state_timer = 0.0

    def on_step(self, sim: Any, dt: float) -> str | None:
        # 1. Main Match Timer (Only counts down while ball is in active play)
        if self.time_limit > 0 and self.state != "GOAL_SCORED":
            self.time_remaining = max(0.0, self.time_remaining - dt)

        # 2. Post-Goal Celebration State (Frozen timer, resets to Kickoff)
        if self.state == "GOAL_SCORED":
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                sim.reset_positions()
                self.state = "KICKOFF"
                self.kickoff_timer = 0.0
                return None

        # 3. Kickoff State
        elif self.state == "KICKOFF":
            # Transition to active play once ball moves from center spot
            if sim.ball.vel.length_sq() > 1.0 or sim.ball.pos.distance_to(sim.center) > 2.0:
                self.state = "PLAYING"
                self.kickoff_timer = 0.0
            else:
                # Accumulate idle kickoff time
                self.kickoff_timer += dt
                if self.kickoff_timer >= self.kickoff_timeout:
                    # Forfeit kickoff right to opponent and reset positions
                    self.kickoff_team = "blue" if self.kickoff_team == "red" else "red"
                    self.kickoff_timer = 0.0
                    sim.reset_positions()

        return None

    def enforce_player_bounds(self, player: Any, sim: Any):
        p = sim.pitch

        # 1. Outer Arena Physical Walls
        if player.pos.x - player.radius < p.outer_left:
            player.pos.x = p.outer_left + player.radius
            player.vel.x = 0
        elif player.pos.x + player.radius > p.outer_right:
            player.pos.x = p.outer_right - player.radius
            player.vel.x = 0

        if player.pos.y - player.radius < p.outer_top:
            player.pos.y = p.outer_top + player.radius
            player.vel.y = 0
        elif player.pos.y + player.radius > p.outer_bottom:
            player.pos.y = p.outer_bottom - player.radius
            player.vel.y = 0

        # 2. Kickoff Circle & Half-Pitch Geometric Boundaries
        if self.state == "KICKOFF":
            R = p.cfg.CENTER_CIRCLE_RADIUS
            r = player.radius
            sign = 1.0 if player.team == "red" else -1.0
            is_attacking = player.team == self.kickoff_team

            rel_x = (player.pos.x - sim.center.x) * sign
            rel_y = player.pos.y - sim.center.y
            vel_x = player.vel.x * sign
            vel_y = player.vel.y

            if is_attacking:
                # Attacking team: Allowed in own half + inside center circle
                if rel_x > 0:
                    dist = math.hypot(rel_x, rel_y)
                    max_dist = R - r
                    if dist > max_dist and dist > 0:
                        nx, ny = rel_x / dist, rel_y / dist
                        rel_x = nx * max_dist
                        rel_y = ny * max_dist
                        v_norm = vel_x * nx + vel_y * ny
                        if v_norm > 0:
                            vel_x -= v_norm * nx
                            vel_y -= v_norm * ny
                else:
                    if abs(rel_y) > R:
                        if rel_x + r > 0:
                            rel_x = -r
                            if vel_x > 0:
                                vel_x = 0
                    else:
                        for corner_y in (-R, R):
                            cdist = math.hypot(rel_x, rel_y - corner_y)
                            if cdist < r and cdist > 0:
                                cnx, cny = rel_x / cdist, (rel_y - corner_y) / cdist
                                rel_x, rel_y = cnx * r, corner_y + cny * r
                                cv_norm = vel_x * cnx + vel_y * cny
                                if cv_norm < 0:
                                    vel_x -= cv_norm * cnx
                                    vel_y -= cv_norm * cny
            else:
                # Defending team: Must stay in own half outside center circle
                if rel_x + r > 0:
                    rel_x = -r
                    if vel_x > 0:
                        vel_x = 0
                dist = math.hypot(rel_x, rel_y)
                min_dist = R + r
                if dist < min_dist and dist > 0:
                    nx, ny = rel_x / dist, rel_y / dist
                    rel_x, rel_y = nx * min_dist, ny * min_dist
                    v_norm = vel_x * nx + vel_y * ny
                    if v_norm < 0:
                        vel_x -= v_norm * nx
                        vel_y -= v_norm * ny

            player.pos.x = sim.center.x + rel_x * sign
            player.pos.y = sim.center.y + rel_y
            player.vel.x = vel_x * sign
            player.vel.y = vel_y

    def enforce_ball_bounds(self, sim: Any) -> str | None:
        p = sim.pitch
        b = sim.ball
        goal_event = None

        # Top and Bottom Touchlines
        if b.pos.y - b.radius < p.top:
            b.pos.y = p.top + b.radius
            b.vel.y *= -b.restitution
        elif b.pos.y + b.radius > p.bottom:
            b.pos.y = p.bottom - b.radius
            b.vel.y *= -b.restitution

        # Left Net & Goal Posts (Red Goal line / Blue scores)
        if b.pos.x - b.radius < p.left:
            if p.goal_top <= b.pos.y <= p.goal_bottom:
                if b.pos.x - b.radius < p.left - p.cfg.GOAL_DEPTH:
                    b.pos.x = p.left - p.cfg.GOAL_DEPTH + b.radius
                    b.vel.x *= -b.restitution
                if b.pos.y - b.radius < p.goal_top:
                    b.pos.y = p.goal_top + b.radius
                    b.vel.y *= -b.restitution
                elif b.pos.y + b.radius > p.goal_bottom:
                    b.pos.y = p.goal_bottom - b.radius
                    b.vel.y *= -b.restitution

                if self.state != "GOAL_SCORED" and b.pos.x < p.left:
                    sim.score_blue += 1
                    self.state = "GOAL_SCORED"
                    self.state_timer = 1.0
                    self.kickoff_team = "red"
                    goal_event = "blue_goal"
            else:
                b.pos.x = p.left + b.radius
                b.vel.x *= -b.restitution

        # Right Net & Goal Posts (Blue Goal line / Red scores)
        elif b.pos.x + b.radius > p.right:
            if p.goal_top <= b.pos.y <= p.goal_bottom:
                if b.pos.x + b.radius > p.right + p.cfg.GOAL_DEPTH:
                    b.pos.x = p.right + p.cfg.GOAL_DEPTH - b.radius
                    b.vel.x *= -b.restitution
                if b.pos.y - b.radius < p.goal_top:
                    b.pos.y = p.goal_top + b.radius
                    b.vel.y *= -b.restitution
                elif b.pos.y + b.radius > p.goal_bottom:
                    b.pos.y = p.goal_bottom - b.radius
                    b.vel.y *= -b.restitution

                if self.state != "GOAL_SCORED" and b.pos.x > p.right:
                    sim.score_red += 1
                    self.state = "GOAL_SCORED"
                    self.state_timer = 1.0
                    self.kickoff_team = "blue"
                    goal_event = "red_goal"
            else:
                b.pos.x = p.right - b.radius
                b.vel.x *= -b.restitution

        return goal_event

    def is_game_over(self, sim: Any) -> bool:
        if self.score_limit > 0:
            if sim.score_red >= self.score_limit or sim.score_blue >= self.score_limit:
                return True
        if self.time_limit > 0 and self.time_remaining <= 0:
            return True
        return False