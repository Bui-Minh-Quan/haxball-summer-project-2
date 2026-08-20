import math
from typing import Any
import pygame
import pygame.gfxdraw
from config.match_config import MatchConfig
from src.engine.simulation import Simulation
from src.game.camera import Camera
from src.game.controllers import KeyboardController
from src.game.state_manager import GameState
from src.game.states.pause_state import PauseState
from src.game.ui.button import Button


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
        self.font_player_num = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_big = pygame.font.SysFont("Arial", 60, bold=True)
        self.font_modal_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_modal_sub = pygame.font.SysFont("Arial", 18)
        self.font_btn = pygame.font.SysFont("Arial", 16, bold=True)

        self.btn_pause = Button(
            pygame.Rect(context.screen_width - 100, 12, 80, 32),
            "Pause",
            self.font_hud,
            self._pause_game,
        )

        self.is_game_over = False
        self._init_game_over_ui()

    def _init_game_over_ui(self):
        cx = self.context.screen_width // 2
        cy = self.context.screen_height // 2
        btn_w, btn_h, gap = 110, 42, 16
        total_w = (btn_w * 3) + (gap * 2)
        start_x = cx - (total_w // 2)
        btn_y = cy + 60

        self.btn_retry = Button(
            pygame.Rect(start_x, btn_y, btn_w, btn_h),
            "Retry",
            self.font_btn,
            self._retry_game,
        )
        self.btn_menu = Button(
            pygame.Rect(start_x + btn_w + gap, btn_y, btn_w, btn_h),
            "Menu",
            self.font_btn,
            self._go_to_menu,
        )
        self.btn_exit = Button(
            pygame.Rect(start_x + (btn_w + gap) * 2, btn_y, btn_w, btn_h),
            "Exit",
            self.font_btn,
            self._exit_game,
        )

    def _pause_game(self):
        if not self.is_game_over:
            self.context.state_manager.push_state(PauseState(self.context))

    def _retry_game(self):
        self.context.state_manager.change_state(PlayState(self.context, self.match_config))

    def _go_to_menu(self):
        from src.game.states.menu_state import MenuState
        self.context.state_manager.change_state(MenuState(self.context))

    def _exit_game(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    def handle_event(self, event: pygame.event.Event):
        if self.is_game_over:
            self.btn_retry.handle_event(event)
            self.btn_menu.handle_event(event)
            self.btn_exit.handle_event(event)
            return

        if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_p):
            self._pause_game()
        self.btn_pause.handle_event(event)

    def update(self, dt: float):
        mode = self.match_config.mode
        if hasattr(mode, "is_game_over") and mode.is_game_over(self.sim):
            self.is_game_over = True

        if not self.is_game_over:
            self.sim.step(dt=dt)

        track_pos = self.sim.red_team[0].pos if self.sim.red_team else self.sim.center
        cam_target = (track_pos * 0.6) + (self.sim.ball.pos * 0.4)

        p = self.sim.pitch
        world_bounds = pygame.Rect(
            p.outer_left, p.outer_top,
            p.outer_right - p.outer_left, p.outer_bottom - p.outer_top,
        )
        self.camera.update(cam_target, world_bounds, dt)

    def _draw_player_halo(self, surface: pygame.Surface, pos: tuple[int, int], radius: int):
        """Draws the translucent glowing ring around the controlled player."""
        halo_radius = radius + 6
        halo_surf = pygame.Surface((halo_radius * 2 + 4, halo_radius * 2 + 4), pygame.SRCALPHA)
        center = (halo_radius + 2, halo_radius + 2)

        # Translucent filled ring
        pygame.draw.circle(halo_surf, (255, 255, 255, 60), center, halo_radius)
        # Outer border
        pygame.draw.circle(halo_surf, (255, 255, 255, 150), center, halo_radius, width=2)

        surface.blit(halo_surf, (pos[0] - halo_radius - 2, pos[1] - halo_radius - 2))

    def _draw_aa_entity(self, surface: pygame.Surface, pos: tuple[int, int], radius: int, fill_color, outline_color, outline_thickness=3):
        x, y = int(pos[0]), int(pos[1])
        pygame.gfxdraw.filled_circle(surface, x, y, radius, outline_color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, outline_color)
        inner_radius = radius - outline_thickness
        if inner_radius > 0:
            pygame.gfxdraw.filled_circle(surface, x, y, inner_radius, fill_color)
            pygame.gfxdraw.aacircle(surface, x, y, inner_radius, fill_color)

    def _draw_kickoff_countdown(self, surface: pygame.Surface, mode: Any):
        if not (hasattr(mode, "state") and mode.state == "KICKOFF"):
            return

        timeout = getattr(mode, "kickoff_timeout", 10.0)
        timer = getattr(mode, "kickoff_timer", 0.0)
        rem_time = max(0.0, timeout - timer)
        progress = rem_time / timeout if timeout > 0 else 0.0
        team = getattr(mode, "kickoff_team", "red")
        team_color = (225, 55, 55) if team == "red" else (50, 110, 235)

        cx = self.context.screen_width // 2
        cy = 82
        badge_w, badge_h = 240, 36
        badge_rect = pygame.Rect(cx - badge_w // 2, cy - badge_h // 2, badge_w, badge_h)

        pygame.draw.rect(surface, (18, 20, 28), badge_rect, border_radius=18)
        pygame.draw.rect(surface, (50, 55, 70), badge_rect, width=2, border_radius=18)

        # Countdown Progress Arc
        arc_cx, arc_cy = cx - (badge_w // 2) + 22, cy
        arc_r = 11
        pygame.gfxdraw.aacircle(surface, arc_cx, arc_cy, arc_r, (60, 65, 80))
        if progress > 0:
            points = [(arc_cx, arc_cy)]
            steps = 30
            for i in range(int(steps * progress) + 1):
                angle = -math.pi / 2 + (2 * math.pi * (i / steps))
                px = arc_cx + math.cos(angle) * arc_r
                py = arc_cy + math.sin(angle) * arc_r
                points.append((px, py))
            if len(points) > 2:
                pygame.draw.polygon(surface, team_color, points)
        pygame.gfxdraw.filled_circle(surface, arc_cx, arc_cy, 6, (18, 20, 28))

        txt = self.font_hud.render(f"{team.upper()} KICK-OFF: {rem_time:.1f}s", True, (240, 240, 240))
        surface.blit(txt, (arc_cx + 18, cy - txt.get_height() // 2))

    def _draw_game_over_modal(self, surface: pygame.Surface):
        overlay = pygame.Surface((self.context.screen_width, self.context.screen_height), pygame.SRCALPHA)
        overlay.fill((10, 12, 18, 200))
        surface.blit(overlay, (0, 0))

        cx = self.context.screen_width // 2
        cy = self.context.screen_height // 2
        card_w, card_h = 480, 280
        card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)

        pygame.draw.rect(surface, (24, 27, 36), card_rect, border_radius=14)
        pygame.draw.rect(surface, (60, 68, 88), card_rect, width=2, border_radius=14)

        r_score = self.sim.score_red
        b_score = self.sim.score_blue
        mode = self.match_config.mode

        if hasattr(mode, "score_limit") and mode.score_limit > 0 and (r_score >= mode.score_limit or b_score >= mode.score_limit):
            sub_text = f"Score limit of {mode.score_limit} reached."
        elif hasattr(mode, "time_remaining") and mode.time_remaining <= 0:
            sub_text = "Full time reached."
        else:
            sub_text = "Match concluded."

        if r_score > b_score:
            title_text, title_color = "RED WINS", (235, 75, 75)
        elif b_score > r_score:
            title_text, title_color = "BLUE WINS", (75, 140, 245)
        else:
            title_text, title_color = "DRAW", (210, 215, 225)

        t_surf = self.font_modal_title.render(title_text, True, title_color)
        surface.blit(t_surf, t_surf.get_rect(center=(cx, cy - 80)))

        score_str = f"RED {r_score}   -   {b_score} BLUE"
        s_surf = self.font_score.render(score_str, True, (240, 240, 240))
        surface.blit(s_surf, s_surf.get_rect(center=(cx, cy - 35)))

        sub_surf = self.font_modal_sub.render(sub_text, True, (150, 160, 180))
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 5)))

        self.btn_retry.draw(surface)
        self.btn_menu.draw(surface)
        self.btn_exit.draw(surface)

    def draw(self, surface: pygame.Surface):
        surface.fill((28, 30, 38))
        p = self.sim.pitch
        cam = self.camera

        # 1. Pitch Markings
        pitch_rect = cam.apply_rect(pygame.Rect(p.left, p.top, p.width, p.height))
        pygame.draw.rect(surface, (45, 125, 60), pitch_rect)
        pygame.draw.rect(surface, (240, 240, 240), pitch_rect, width=4)

        pygame.draw.line(
            surface, (240, 240, 240),
            cam.apply((p.center.x, p.top)),
            cam.apply((p.center.x, p.bottom)), 2
        )
        cx, cy = cam.apply((p.center.x, p.center.y))
        c_rad = int(p.cfg.CENTER_CIRCLE_RADIUS)
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad, (240, 240, 240))
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad - 1, (240, 240, 240))

        # 2. Goal Nets
        left_goal = cam.apply_rect(pygame.Rect(p.left - p.cfg.GOAL_DEPTH, p.goal_top, p.cfg.GOAL_DEPTH, p.cfg.GOAL_HEIGHT))
        right_goal = cam.apply_rect(pygame.Rect(p.right, p.goal_top, p.cfg.GOAL_DEPTH, p.cfg.GOAL_HEIGHT))
        pygame.draw.rect(surface, (35, 95, 45), left_goal)
        pygame.draw.rect(surface, (240, 240, 240), left_goal, width=3)
        pygame.draw.rect(surface, (35, 95, 45), right_goal)
        pygame.draw.rect(surface, (240, 240, 240), right_goal, width=3)

        for post in p.posts:
            self._draw_aa_entity(surface, cam.apply(post.pos), int(post.radius), (255, 255, 255), (40, 40, 40), 1)

        # 3. Ball
        ball = self.sim.ball
        self._draw_aa_entity(surface, cam.apply(ball.pos), int(ball.radius), (255, 255, 255), (20, 20, 20), 2)

        # 4. Players (with Controlled Halo Indicator & Numbers)
        for i, player in enumerate(self.sim.red_team):
            pos = cam.apply(player.pos)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)

            # Check if this player is controlled by keyboard
            slot = self.match_config.roster[i] if i < len(self.match_config.roster) else None
            is_human = isinstance(getattr(slot, "controller", None), KeyboardController)

            if is_human:
                self._draw_player_halo(surface, pos, int(player.radius))

            self._draw_aa_entity(surface, pos, int(player.radius), (225, 55, 55), out_c, 3)

            # Number Text
            num_txt = self.font_player_num.render(str(i + 1), True, (255, 255, 255))
            surface.blit(num_txt, num_txt.get_rect(center=pos))

        for j, player in enumerate(self.sim.blue_team):
            pos = cam.apply(player.pos)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)

            self._draw_aa_entity(surface, pos, int(player.radius), (50, 110, 235), out_c, 3)

            num_txt = self.font_player_num.render(str(j + 1), True, (255, 255, 255))
            surface.blit(num_txt, num_txt.get_rect(center=pos))

        # 5. Off-Screen Ball Indicator
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

        # 6. In-Game Goal Banner
        mode = self.match_config.mode
        if hasattr(mode, "state") and mode.state == "GOAL_SCORED" and not self.is_game_over:
            goal_txt = self.font_big.render("GOAL!", True, (255, 220, 50))
            surface.blit(goal_txt, goal_txt.get_rect(center=(self.context.screen_width // 2, self.context.screen_height // 2 - 100)))

        # 7. Kickoff Timer
        if not self.is_game_over:
            self._draw_kickoff_countdown(surface, mode)

        # 8. Top HUD Bar
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

        if hasattr(mode, "time_remaining"):
            mins = int(mode.time_remaining) // 60
            secs = int(mode.time_remaining) % 60
            timer_surf = self.font_score.render(f"{mins:02d}:{secs:02d}", True, (220, 220, 220))
            surface.blit(timer_surf, (20, 10))

        if not self.is_game_over:
            self.btn_pause.draw(surface)

        # 9. Game Over Modal Overlay
        if self.is_game_over:
            self._draw_game_over_modal(surface)