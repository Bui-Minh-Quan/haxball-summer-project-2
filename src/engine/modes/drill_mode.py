from typing import Any
from src.engine.modes.base_mode import GameMode
from src.engine.vector import Vec2


class SoloDrillMode(GameMode):
    """Curriculum Stage 1: Solo shooting drill."""

    def __init__(self, time_limit: float = 0.0):
        self.time_limit = time_limit

    def init_mode(self, sim: Any):
        sim.score_red = 0
        sim.score_blue = 0

    def on_step(self, sim: Any, dt: float) -> str | None:
        return None

    def enforce_player_bounds(self, player: Any, sim: Any):
        p = sim.pitch
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

    def enforce_ball_bounds(self, sim: Any) -> str | None:
        p = sim.pitch
        b = sim.ball
        goal_event = None

        # Top and Bottom walls
        if b.pos.y - b.radius < p.top:
            b.pos.y = p.top + b.radius
            b.vel.y *= -b.restitution
        elif b.pos.y + b.radius > p.bottom:
            b.pos.y = p.bottom - b.radius
            b.vel.y *= -b.restitution

        # Left Boundary (Own Goal)
        if b.pos.x - b.radius < p.left:
            if p.goal_top <= b.pos.y <= p.goal_bottom:
                if b.pos.x < p.left:
                    sim.score_blue += 1
                    goal_event = "blue_goal"
            else:
                b.pos.x = p.left + b.radius
                b.vel.x *= -b.restitution

        # Right Boundary (Target Goal)
        elif b.pos.x + b.radius > p.right:
            if p.goal_top <= b.pos.y <= p.goal_bottom:
                if b.pos.x > p.right:
                    sim.score_red += 1
                    goal_event = "red_goal"
            else:
                b.pos.x = p.right - b.radius
                b.vel.x *= -b.restitution

        return goal_event

    def is_game_over(self, sim: Any) -> bool:
        return False