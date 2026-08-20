import math
from typing import Any

from config.match_config import MatchConfig
from config.physics_config import PhysicsConfig
from src.engine.entities import Ball, Disc, Player
from src.engine.pitch import Pitch
from src.engine.vector import Vec2


class Simulation:

    def __init__(
        self,
        center_x: float = 600.0,
        center_y: float = 400.0,
        match_config: MatchConfig | None = None,
        cfg: PhysicsConfig | None = None,
    ):
        self.cfg = cfg or PhysicsConfig()
        self.match_config = match_config or MatchConfig()

        self.mode = self.match_config.mode
        if self.mode:
            self.mode.init_mode(self)

        self.center = Vec2(center_x, center_y)

        # Dynamic Pitch Dimensions
        pw = match_config.pitch_width if match_config else self.cfg.DEFAULT_PITCH_WIDTH
        ph = match_config.pitch_height if match_config else self.cfg.DEFAULT_PITCH_HEIGHT
        self.pitch = Pitch(self.center, self.cfg, width=pw, height=ph)

        self.ball = Ball(self.center, self.cfg)
        self.all_players: list[Player] = []
        self.controllers: list[Any] = []
        self.red_team: list[Player] = []
        self.blue_team: list[Player] = []

        self.score_red = 0
        self.score_blue = 0

        self._spawn_roster()
        if self.match_config and self.match_config.mode:
            self.match_config.mode.init_mode(self)

    def _spawn_roster(self):
        if not self.match_config or not self.match_config.roster:
            return

        red_slots = [s for s in self.match_config.roster if s.team == "red"]
        blue_slots = [s for s in self.match_config.roster if s.team == "blue"]

        # Spawn Red team
        y_step_red = self.pitch.height / (len(red_slots) + 1)
        for i, slot in enumerate(red_slots):
            y_pos = self.pitch.top + y_step_red * (i + 1)
            player = Player(Vec2(self.pitch.left + 200, y_pos), "red", slot.stats)
            self.red_team.append(player)
            self.all_players.append(player)
            self.controllers.append(slot.controller)

        # Spawn Blue team
        y_step_blue = self.pitch.height / (len(blue_slots) + 1)
        for i, slot in enumerate(blue_slots):
            y_pos = self.pitch.top + y_step_blue * (i + 1)
            player = Player(Vec2(self.pitch.right - 200, y_pos), "blue", slot.stats)
            self.blue_team.append(player)
            self.all_players.append(player)
            self.controllers.append(slot.controller)

    def reset_positions(self):
        self.ball.pos = self.center.copy()
        self.ball.vel = Vec2(0.0, 0.0)

        y_step_red = self.pitch.height / (len(self.red_team) + 1)
        for i, player in enumerate(self.red_team):
            player.pos = Vec2(self.pitch.left + 200, self.pitch.top + y_step_red * (i + 1))
            player.vel = Vec2(0.0, 0.0)

        y_step_blue = self.pitch.height / (len(self.blue_team) + 1)
        for i, player in enumerate(self.blue_team):
            player.pos = Vec2(self.pitch.right - 200, self.pitch.top + y_step_blue * (i + 1))
            player.vel = Vec2(0.0, 0.0)

    def step(self, dt: float) -> str | None:
        if self.match_config:
            dt *= self.match_config.game_speed

        mode = self.match_config.mode if self.match_config else None

        # 1. Mode Hook
        if mode:
            mode.on_step(self, dt)

        # 2. Query Controllers
        for idx, (player, controller) in enumerate(zip(self.all_players, self.controllers)):
            move, kick = controller.get_action(idx, self)
            player.process_input(move, kick, dt)

        # 3. Micro-Stepping
        SUBSTEPS = 6
        sub_dt = dt / SUBSTEPS
        goal_event = None

        for _ in range(SUBSTEPS):
            for player in self.all_players:
                player.apply_accel(sub_dt)
                player.step(sub_dt)
            self.ball.step(sub_dt)

            for player in self.all_players:
                self._resolve_kick(player)

            # Player vs Player Collisions (Dynamic restitution)
            players = self.all_players
            num_players = len(players)
            for i in range(num_players):
                for j in range(i + 1, num_players):
                    combined_restitution = math.sqrt(players[i].restitution * players[j].restitution)
                    self._resolve_circle_collision(players[i], players[j], combined_restitution)

            # Player vs Ball Collisions (Soft control restitution)
            for player in self.all_players:
                self._resolve_circle_collision(player, self.ball, self.cfg.BALL_PLAYER_RESTITUTION)

            # Post Collisions
            for post in self.pitch.posts:
                self._resolve_circle_collision(self.ball, post, self.cfg.BALL_RESTITUTION)
                for player in self.all_players:
                    self._resolve_circle_collision(player, post, player.restitution)

            # Mode Enforced Boundaries
            if mode:
                for player in self.all_players:
                    mode.enforce_player_bounds(player, self)
                evt = mode.enforce_ball_bounds(self)
                if evt:
                    goal_event = evt

        return goal_event

    def _resolve_kick(self, player: Player):
        if not player.is_kicking:
            return

        to_ball = self.ball.pos - player.pos
        dist = to_ball.length()
        kick_reach = player.radius + self.ball.radius + player.stats.kick_margin

        if 0 < dist <= kick_reach:
            normal = to_ball.normalize()
            self.ball.vel = player.vel + (normal * player.stats.kick_strength)
            player.is_kicking = False

    def _resolve_circle_collision(self, d1: Disc, d2: Disc, restitution: float):
        delta = d2.pos - d1.pos
        dist = delta.length()
        min_dist = d1.radius + d2.radius

        if 0 < dist < min_dist:
            normal = delta / dist
            overlap = min_dist - dist
            total_inv = d1.inv_mass + d2.inv_mass
            if total_inv > 0:
                d1.pos -= normal * (overlap * (d1.inv_mass / total_inv))
                d2.pos += normal * (overlap * (d2.inv_mass / total_inv))

            rel_vel = d1.vel - d2.vel
            vel_along_normal = rel_vel.dot(normal)
            if vel_along_normal > 0:
                j = -(1.0 + restitution) * vel_along_normal
                j /= total_inv
                impulse = normal * j
                d1.vel += impulse * d1.inv_mass
                d2.vel -= impulse * d2.inv_mass