from config.physics_config import PhysicsConfig
from src.engine.entities import Ball, Disc, Player
from src.engine.pitch import Pitch
from src.engine.vector import Vec2
import math

class Simulation:

    def __init__(self, center_x: float = 600.0, center_y: float = 400.0, team_size: int = 1, cfg: PhysicsConfig | None = None):
        self.cfg = cfg or PhysicsConfig()
        self.center = Vec2(center_x, center_y)
        self.pitch = Pitch(self.center, self.cfg)
        self.team_size = team_size

        self.ball = Ball(self.center, self.cfg)
        self.red_team: list[Player] = []
        self.blue_team: list[Player] = []
        self._spawn_teams()

        self.score_red = 0
        self.score_blue = 0
        
        # Match States: "KICKOFF", "PLAYING", "GOAL_SCORED"
        self.state = "KICKOFF"
        self.kickoff_team = "red"  # Who gets the ball first
        self.state_timer = 0.0

    @property
    def all_players(self) -> list[Player]:
        return self.red_team + self.blue_team

    def _spawn_teams(self):
        self.red_team.clear()
        self.blue_team.clear()
        y_step = self.cfg.PITCH_HEIGHT / (self.team_size + 1)
        for i in range(self.team_size):
            y_pos = self.pitch.top + y_step * (i + 1)
            self.red_team.append(Player(Vec2(self.pitch.left + 200, y_pos), "red", self.cfg))
            self.blue_team.append(Player(Vec2(self.pitch.right - 200, y_pos), "blue", self.cfg))

    def reset_positions(self):
        self.ball.pos = self.center.copy()
        self.ball.vel = Vec2(0.0, 0.0)

        y_step = self.cfg.PITCH_HEIGHT / (self.team_size + 1)
        for i, player in enumerate(self.red_team):
            player.pos = Vec2(self.pitch.left + 200, self.pitch.top + y_step * (i + 1))
            player.vel = Vec2(0.0, 0.0)

        for i, player in enumerate(self.blue_team):
            player.pos = Vec2(self.pitch.right - 200, self.pitch.top + y_step * (i + 1))
            player.vel = Vec2(0.0, 0.0)
            
        self.state = "KICKOFF"

    def step(self, red_inputs: list[tuple[Vec2, bool]], blue_inputs: list[tuple[Vec2, bool]], dt: float) -> str | None:
        dt *= self.cfg.GAME_SPEED
        
        # 1. State Machine Overrides
        if self.state == "GOAL_SCORED":
            red_inputs = [(Vec2(0,0), False) for _ in self.red_team]
            blue_inputs = [(Vec2(0,0), False) for _ in self.blue_team]
            for p in self.all_players:
                p.vel *= 0.8
            
            self.state_timer -= dt
            if self.state_timer <= 0:
                self.reset_positions()
                return None

        elif self.state == "KICKOFF":
            if self.ball.vel.length_sq() > 0 or self.ball.pos.distance_to(self.center) > 2.0:
                self.state = "PLAYING"

        # 2. PROCESS INPUTS ONCE PER FRAME (Outside the substep loop)
        for player, inp in zip(self.red_team, red_inputs):
            player.process_input(inp[0], inp[1], dt)
        for player, inp in zip(self.blue_team, blue_inputs):
            player.process_input(inp[0], inp[1], dt)

        # 3. Physics Micro-stepping
        SUBSTEPS = 6
        sub_dt = dt / SUBSTEPS
        triggered_goal = None
        
        for _ in range(SUBSTEPS):
            for player in self.all_players:
                player.apply_accel(sub_dt)
                player.step(sub_dt)
            self.ball.step(sub_dt)

            for player in self.all_players:
                self._resolve_kick(player)

            players = self.all_players
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    self._resolve_circle_collision(players[i], players[j], self.cfg.PLAYER_RESTITUTION)

            for player in self.all_players:
                self._resolve_circle_collision(player, self.ball, self.cfg.BALL_PLAYER_RESTITUTION)

            for post in self.pitch.posts:
                self._resolve_circle_collision(self.ball, post, self.cfg.BALL_RESTITUTION)
                for player in self.all_players:
                    self._resolve_circle_collision(player, post, self.cfg.PLAYER_RESTITUTION)

            for player in self.all_players:
                self._resolve_player_bounds(player)
            
            evt = self._resolve_ball_bounds()
            if evt:
                triggered_goal = evt

        return triggered_goal


    def _resolve_kick(self, player: Player):
        if not player.is_kicking:
            return

        to_ball = self.ball.pos - player.pos
        dist = to_ball.length()
        kick_reach = player.radius + self.ball.radius + self.cfg.KICK_MARGIN

        if dist <= kick_reach and dist > 0:
            normal = to_ball.normalize()
            effective_kick = self.cfg.KICK_STRENGTH * self.cfg.BALL_SPEED_MULT
            
            # ADD player's current velocity to the kick for powerful forward shots
            self.ball.vel = player.vel + (normal * effective_kick)
            
            # Consume the kick so it doesn't hit again in the next micro-substep
            player.is_kicking = False


    def _resolve_circle_collision(self, d1: Disc, d2: Disc, restitution: float):
        delta = d2.pos - d1.pos
        dist = delta.length()
        min_dist = d1.radius + d2.radius

        if dist < min_dist and dist > 0:
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

    def _resolve_player_bounds(self, player: Player):
        p = self.pitch

        # 1. Outer Arena Fences
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

        # 2. Kick-off Continuous Geometry
        if self.state == "KICKOFF":
            R = self.cfg.CENTER_CIRCLE_RADIUS
            r = player.radius
            
            # sign = +1 for Red (attacks right), -1 for Blue (attacks left)
            sign = 1.0 if player.team == "red" else -1.0
            is_attacking = (player.team == self.kickoff_team)

            # Local coordinates: positive rel_x = opponent's half, negative rel_x = own half
            rel_x = (player.pos.x - self.center.x) * sign
            rel_y = player.pos.y - self.center.y
            vel_x = player.vel.x * sign
            vel_y = player.vel.y

            if is_attacking:
                # ---------------------------------------------------------
                # ATTACKING TEAM (Own half + entire center circle)
                # ---------------------------------------------------------
                if rel_x > 0:
                    # Inside opponent's half: must stay within circle radius R
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
                    # Inside own half: cannot cross midline outside circle (|rel_y| > R)
                    if abs(rel_y) > R:
                        if rel_x + r > 0:
                            rel_x = -r
                            if vel_x > 0:
                                vel_x = 0
                    else:
                        # Smooth corner junctions at (0, -R) and (0, +R)
                        for corner_y in (-R, R):
                            cdist = math.hypot(rel_x, rel_y - corner_y)
                            if cdist < r and cdist > 0:
                                cnx = rel_x / cdist
                                cny = (rel_y - corner_y) / cdist
                                rel_x = cnx * r
                                rel_y = corner_y + cny * r
                                cv_norm = vel_x * cnx + vel_y * cny
                                if cv_norm < 0:
                                    vel_x -= cv_norm * cnx
                                    vel_y -= cv_norm * cny

            else:
                # ---------------------------------------------------------
                # DEFENDING TEAM (Must stay strictly in own half: rel_x <= -r)
                # ---------------------------------------------------------
                # A. Cannot cross midline into opponent's territory
                if rel_x + r > 0:
                    rel_x = -r
                    if vel_x > 0:
                        vel_x = 0

                # B. Cannot enter center circle (stay outside radius R)
                dist = math.hypot(rel_x, rel_y)
                min_dist = R + r
                if dist < min_dist and dist > 0:
                    nx, ny = rel_x / dist, rel_y / dist
                    rel_x = nx * min_dist
                    rel_y = ny * min_dist
                    v_norm = vel_x * nx + vel_y * ny
                    if v_norm > 0:
                        vel_x -= v_norm * nx
                        vel_y -= v_norm * ny

            # Transform back to global coordinates
            player.pos.x = self.center.x + rel_x * sign
            player.pos.y = self.center.y + rel_y
            player.vel.x = vel_x * sign
            player.vel.y = vel_y


    def _resolve_ball_bounds(self) -> str | None:
        p = self.pitch
        b = self.ball
        goal_event = None

        # 1. Pitch Top and Bottom
        if b.pos.y - b.radius < p.top:
            b.pos.y = p.top + b.radius
            b.vel.y *= -b.restitution
        elif b.pos.y + b.radius > p.bottom:
            b.pos.y = p.bottom - b.radius
            b.vel.y *= -b.restitution

        # 2. Left Boundary & Goal Net
        if b.pos.x - b.radius < p.left:
            if p.goal_top <= b.pos.y <= p.goal_bottom:
                # Inner Goal Net Bouncing
                if b.pos.x - b.radius < p.left - self.cfg.GOAL_DEPTH:
                    b.pos.x = p.left - self.cfg.GOAL_DEPTH + b.radius
                    b.vel.x *= -b.restitution
                
                if b.pos.y - b.radius < p.goal_top:
                    b.pos.y = p.goal_top + b.radius
                    b.vel.y *= -b.restitution
                elif b.pos.y + b.radius > p.goal_bottom:
                    b.pos.y = p.goal_bottom - b.radius
                    b.vel.y *= -b.restitution
                
                # Goal Trigger
                if self.state != "GOAL_SCORED" and b.pos.x < p.left:
                    self.score_blue += 1
                    self.state = "GOAL_SCORED"
                    self.state_timer = 1.0  # 1 second celebration
                    self.kickoff_team = "red" # Conceding team gets kick-off
                    goal_event = "blue_goal"
            else:
                b.pos.x = p.left + b.radius
                b.vel.x *= -b.restitution

        # 3. Right Boundary & Goal Net
        elif b.pos.x + b.radius > p.right:
            if p.goal_top <= b.pos.y <= p.goal_bottom:
                # Inner Goal Net Bouncing
                if b.pos.x + b.radius > p.right + self.cfg.GOAL_DEPTH:
                    b.pos.x = p.right + self.cfg.GOAL_DEPTH - b.radius
                    b.vel.x *= -b.restitution
                
                if b.pos.y - b.radius < p.goal_top:
                    b.pos.y = p.goal_top + b.radius
                    b.vel.y *= -b.restitution
                elif b.pos.y + b.radius > p.goal_bottom:
                    b.pos.y = p.goal_bottom - b.radius
                    b.vel.y *= -b.restitution
                
                # Goal Trigger
                if self.state != "GOAL_SCORED" and b.pos.x > p.right:
                    self.score_red += 1
                    self.state = "GOAL_SCORED"
                    self.state_timer = 2.0
                    self.kickoff_team = "blue"
                    goal_event = "red_goal"
            else:
                b.pos.x = p.right - b.radius
                b.vel.x *= -b.restitution

        return goal_event