from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2


class HeuristicBot:

    def __init__(self, team: str = "blue"):
        self.team = team

    def get_action(
        self, bot_player: Player, sim: Simulation
    ) -> tuple[Vec2, bool]:
        """
        Calculates movement vector and kick trigger based on pitch geometry.
        """
        ball = sim.ball
        target_goal_x = sim.pitch.left if self.team == "blue" else sim.pitch.right
        own_goal_x = sim.pitch.right if self.team == "blue" else sim.pitch.left

        # Vector from bot to ball
        to_ball = ball.pos - bot_player.pos
        dist_to_ball = to_ball.length()

        # Strategic offset: get behind the ball relative to the target goal
        behind_offset = 25.0 if self.team == "blue" else -25.0
        target_pos = Vec2(ball.pos.x + behind_offset, ball.pos.y)
        target_pos = bot_player.pos
        move_dir = (target_pos - bot_player.pos).normalize()

        # Kick when close and facing the opponent's half
        kick = False
        kick_reach = (
            bot_player.radius + ball.radius + sim.cfg.KICK_MARGIN + 8.0
        )
        if dist_to_ball <= kick_reach:
            if (self.team == "blue" and to_ball.x < 0) or (
                self.team == "red" and to_ball.x > 0
            ):
                kick = True

        return move_dir, kick