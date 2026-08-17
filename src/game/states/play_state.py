import pygame
import pygame.gfxdraw
from config.match_config import MatchConfig
from src.engine.simulation import Simulation
from src.game.camera import Camera
from src.game.state_manager import GameState
from src.game.states.pause_state import PauseState
from src.game.ui.button import Button
import math

class PlayState(GameState):

    def __init__(self, context, match_config: MatchConfig):
        super().__init__(context)
        self.match_config = match_config
        self.sim = Simulation(
            center_x=context.screen_width / 2,
            center_y=context.screen_height / 2 + 20,
            match_config=match_config,
        )

        self.camera = Camera(context.screen_width, context.screen_height, hud_height=50)
        self.font_score = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_hud = pygame.font.SysFont("Arial", 16)
        self.font_big = pygame.font.SysFont("Arial", 60, bold=True)

        self.btn_pause = Button(
            pygame.Rect(context.screen_width - 100, 12, 80, 32),
            "Pause",
            self.font_hud,
            self._pause_game,
        )

    def _pause_game(self):
        self.context.state_manager.push_state(PauseState(self.context))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_p):
            self._pause_game()
        self.btn_pause.handle_event(event)

    def update(self, dt: float):
        self.sim.step(dt=dt)

        # Dual Focus Camera (Tracks first red player & ball)
        track_pos = self.sim.red_team[0].pos if self.sim.red_team else self.sim.center
        cam_target = (track_pos * 0.6) + (self.sim.ball.pos * 0.4)

        p = self.sim.pitch
        world_bounds = pygame.Rect(
            p.outer_left, p.outer_top,
            p.outer_right - p.outer_left, p.outer_bottom - p.outer_top
        )
        self.camera.update(cam_target, world_bounds, dt)

    def _draw_aa_entity(self, surface, pos, radius, fill_color, outline_color, outline_thickness=3):
        x, y = pos
        pygame.gfxdraw.filled_circle(surface, x, y, radius, outline_color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, outline_color)
        inner_radius = radius - outline_thickness
        if inner_radius > 0:
            pygame.gfxdraw.filled_circle(surface, x, y, inner_radius, fill_color)
            pygame.gfxdraw.aacircle(surface, x, y, inner_radius, fill_color)

    def draw(self, surface: pygame.Surface):
        surface.fill((28, 30, 38))
        p = self.sim.pitch
        cam = self.camera

        # 1. Pitch
        pitch_rect = cam.apply_rect(pygame.Rect(p.left, p.top, p.width, p.height))
        pygame.draw.rect(surface, (45, 125, 60), pitch_rect)
        pygame.draw.rect(surface, (240, 240, 240), pitch_rect, width=4)

        # Center line & Circle
        pygame.draw.line(surface, (240, 240, 240), cam.apply((p.center.x, p.top)), cam.apply((p.center.x, p.bottom)), 2)
        cx, cy = cam.apply((p.center.x, p.center.y))
        c_rad = int(p.cfg.CENTER_CIRCLE_RADIUS)
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad, (240, 240, 240))
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad - 1, (240, 240, 240))

        # Goal Nets
        left_goal = cam.apply_rect(pygame.Rect(p.left - p.cfg.GOAL_DEPTH, p.goal_top, p.cfg.GOAL_DEPTH, p.cfg.GOAL_HEIGHT))
        right_goal = cam.apply_rect(pygame.Rect(p.right, p.goal_top, p.cfg.GOAL_DEPTH, p.cfg.GOAL_HEIGHT))
        pygame.draw.rect(surface, (35, 95, 45), left_goal)
        pygame.draw.rect(surface, (240, 240, 240), left_goal, width=3)
        pygame.draw.rect(surface, (35, 95, 45), right_goal)
        pygame.draw.rect(surface, (240, 240, 240), right_goal, width=3)

        for post in p.posts:
            self._draw_aa_entity(surface, cam.apply(post.pos), int(post.radius), (255, 255, 255), (40, 40, 40), 1)

        # Ball
        ball = self.sim.ball
        self._draw_aa_entity(surface, cam.apply(ball.pos), int(ball.radius), (255, 255, 255), (20, 20, 20), 2)

        # Players
        for player in self.sim.red_team:
            pos = cam.apply(player.pos)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)
            self._draw_aa_entity(surface, pos, int(player.radius), (225, 55, 55), out_c, 3)

        for player in self.sim.blue_team:
            pos = cam.apply(player.pos)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)
            self._draw_aa_entity(surface, pos, int(player.radius), (50, 110, 235), out_c, 3)

        # Off-Screen Ball Indicator
        ball_sx, ball_sy = cam.apply(ball.pos)
        margin, hud_h = 25, 50
        if ball_sx < 0 or ball_sx > self.context.screen_width or ball_sy < hud_h or ball_sy > self.context.screen_height:
            cx_scr, cy_scr = self.context.screen_width / 2, (self.context.screen_height + hud_h) / 2
            dx, dy = ball_sx - cx_scr, ball_sy - cy_scr
            dist = math.hypot(dx, dy)
            if dist > 0:
                nx, ny = dx / dist, dy / dist
                ptr_x = max(margin, min(self.context.screen_width - margin, ball_sx))
                ptr_y = max(hud_h + margin, min(self.context.screen_height - margin, ball_sy))
                px_v, py_v = -ny, nx
                p1 = (ptr_x + nx * 12, ptr_y + ny * 12)
                p2 = (ptr_x + px_v * 10 - nx * 12, ptr_y + py_v * 10 - ny * 12)
                p3 = (ptr_x - px_v * 10 - nx * 12, ptr_y - py_v * 10 - ny * 12)
                pygame.gfxdraw.filled_polygon(surface, [p1, p2, p3], (255, 255, 255))
                pygame.gfxdraw.aapolygon(surface, [p1, p2, p3], (20, 20, 20))

        # Goal Notification
        mode = self.match_config.mode
        if hasattr(mode, "state") and mode.state == "GOAL_SCORED":
            goal_txt = self.font_big.render("GOAL!", True, (255, 220, 50))
            surface.blit(goal_txt, goal_txt.get_rect(center=(self.context.screen_width // 2, self.context.screen_height // 2 - 100)))

        # HUD
        hud_bar = pygame.Rect(0, 0, self.context.screen_width, 50)
        pygame.draw.rect(surface, (20, 22, 30), hud_bar)
        pygame.draw.line(surface, (50, 55, 70), (0, 50), (self.context.screen_width, 50), 2)

        red_txt = self.font_score.render(f"RED {self.sim.score_red}", True, (225, 55, 55))
        dash_txt = self.font_score.render("  -  ", True, (245, 245, 245))
        blue_txt = self.font_score.render(f"{self.sim.score_blue} BLUE", True, (50, 110, 235))
        tot_w = red_txt.get_width() + dash_txt.get_width() + blue_txt.get_width()
        start_x = (self.context.screen_width - tot_w) // 2
        surface.blit(red_txt, (start_x, 10))
        surface.blit(dash_txt, (start_x + red_txt.get_width(), 10))
        surface.blit(blue_txt, (start_x + red_txt.get_width() + dash_txt.get_width(), 10))

        # Match Timer
        if hasattr(mode, "time_remaining"):
            mins = int(mode.time_remaining) // 60
            secs = int(mode.time_remaining) % 60
            timer_surf = self.font_score.render(f"{mins:02d}:{secs:02d}", True, (220, 220, 220))
            surface.blit(timer_surf, (20, 10))

        self.btn_pause.draw(surface)