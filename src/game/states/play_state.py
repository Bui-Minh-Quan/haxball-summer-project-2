import math
import os
import random
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

        def exit(self):
            """Stops background music when leaving the match state."""
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.stop()

        self.match_config = match_config
        self.sim = Simulation(
            center_x=context.screen_width / 2,
            center_y=context.screen_height / 2 + 20,
            match_config=match_config,
        )

        self.camera = Camera(context.screen_width, context.screen_height, hud_height=60)

        # UI Fonts
        self.font_score = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_time = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_player_num = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_banner = pygame.font.SysFont("Arial", 68, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 14, bold=True)

        self.last_scorer = None
        self._was_ball_touching = False

        # Match Finish / Victorious Sequence State
        self.is_match_finished = False
        self.end_game_phase = None  # "GOAL_CELEBRATION", "TIMEOUT", "VICTORY"
        self.end_game_timer = 0.0
        self.victor_name = None

        # ── AUDIO INITIALIZATION & PERSISTENCE ──
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Load persisted audio preference (default: True)
        if not hasattr(self.context, "sound_enabled"):
            self.context.sound_enabled = True
        self.sound_enabled = self.context.sound_enabled

        # Celebration Sound
        self.sound_celebration = None
        snd_cel = "assets/sounds/crowd/crowd_celebration.mp3"
        if os.path.exists(snd_cel):
            try:
                self.sound_celebration = pygame.mixer.Sound(snd_cel)
                self.sound_celebration.set_volume(0.65)
            except Exception as e:
                print(f"Warning: Could not load celebration sound: {e}")

        # Ball Kick Sounds
        self.kick_sounds = []
        for path in [
            "assets/sounds/balls/ball_kick1.wav",
            #"assets/sounds/balls/ball_kick2.wav",
            #"assets/sounds/balls/ball_kick3.mp3",
        ]:
            if os.path.exists(path):
                try:
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(0.50)
                    self.kick_sounds.append(snd)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")

        # Looping Ambient Crowd Noise
        bg_sound_path = "assets/sounds/crowd/background_crowd_noise.mp3"
        if os.path.exists(bg_sound_path):
            try:
                pygame.mixer.music.load(bg_sound_path)
                pygame.mixer.music.set_volume(0.40 if self.sound_enabled else 0.0)
                pygame.mixer.music.play(-1)
            except Exception as e:
                print(f"Warning: Could not load background ambient music: {e}")

        # HUD Action Buttons
        self.btn_pause = Button(
            pygame.Rect(context.screen_width - 100, 15, 85, 32),
            "Menu ≡",
            self.font_btn,
            self._pause_game,
            base_color=(45, 55, 75),
            hover_color=(60, 75, 100),
        )

        # Sound Toggle Icon Button
        self.btn_sound_rect = pygame.Rect(context.screen_width - 145, 15, 36, 32)
        self._load_sound_icons()

        # Visual Asset Caches
        self._sprite_cache = {}
        self._load_ball_sprite()

    def _load_sound_icons(self):
        self.icon_volume = None
        self.icon_muted = None
        v_path = "assets/images/sounds/volume.png"
        m_path = "assets/images/sounds/no-sound.png"
        try:
            if os.path.exists(v_path):
                img_v = pygame.image.load(v_path).convert_alpha()
                self.icon_volume = pygame.transform.smoothscale(img_v, (20, 20))
            if os.path.exists(m_path):
                img_m = pygame.image.load(m_path).convert_alpha()
                self.icon_muted = pygame.transform.smoothscale(img_m, (20, 20))
        except Exception as e:
            print(f"Warning: Could not load sound icons: {e}")

    def _load_ball_sprite(self):
        ball_path = "assets/images/balls/ball1.png"
        self.ball_img = None
        diam = int(self.sim.ball.radius * 2)
        if os.path.exists(ball_path):
            try:
                img = pygame.image.load(ball_path).convert_alpha()
                bbox = img.get_bounding_rect()
                cropped = img.subsurface(bbox)
                self.ball_img = pygame.transform.smoothscale(cropped, (diam, diam))
            except Exception as e:
                print(f"Warning: Could not load ball sprite: {e}")

    def _toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        self.context.sound_enabled = self.sound_enabled
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.set_volume(0.20 if self.sound_enabled else 0.0)

    def _play_kick_sound(self):
        if self.sound_enabled and self.kick_sounds:
            random.choice(self.kick_sounds).play()

    def _get_aa_circle(self, radius: int, fill_color, outline_color, outline_thickness=3):
        key = (radius, fill_color, outline_color, outline_thickness)
        if key in self._sprite_cache:
            return self._sprite_cache[key]

        scale = 4
        surf_size = (radius * 2 + outline_thickness * 2) * scale
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = surf_size // 2

        pygame.draw.circle(surf, outline_color, (center, center), radius * scale + outline_thickness * scale)
        pygame.draw.circle(surf, fill_color, (center, center), radius * scale)

        final_size = radius * 2 + outline_thickness * 2
        aa_surf = pygame.transform.smoothscale(surf, (final_size, final_size))
        self._sprite_cache[key] = aa_surf
        return aa_surf

    def _pause_game(self):
        if not self.is_match_finished:
            if pygame.mixer.get_init():
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                pygame.mixer.pause()  # Freezes all active sound effects (celebration, kicks)
            self.context.state_manager.push_state(PauseState(self.context))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_p):
            self._pause_game()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_sound_rect.collidepoint(event.pos):
                self._toggle_sound()
                return

        self.btn_pause.handle_event(event)

    def update(self, dt: float):
        mode = self.match_config.mode

        if not self.is_match_finished:
            # 1. Step simulation
            goal_event = self.sim.step(dt=dt)

            # 2. Trigger kick sound strictly on physical kick impulse
            if getattr(self.sim, "kicked_this_step", False):
                self._play_kick_sound()

            # 3. Check for goal event
            if goal_event is not None:
                self.last_scorer = "Red" if goal_event == "red_goal" else "Blue"
                if self.sound_enabled and self.sound_celebration:
                    self.sound_celebration.play()

                # FLOW A: Goal scored to finish game (score limit reached)
                if mode.score_limit > 0 and (self.sim.score_red >= mode.score_limit or self.sim.score_blue >= mode.score_limit):
                    self.is_match_finished = True
                    self.end_game_phase = "GOAL_CELEBRATION"
                    self.end_game_timer = 2.2  # Step 1: Announce "[Team] Scores!"
                    self.victor_name = "Red" if self.sim.score_red > self.sim.score_blue else "Blue"

            # FLOW B: Time expired without a finishing goal
            elif hasattr(mode, "time_remaining") and mode.time_limit > 0 and mode.time_remaining <= 0:
                self.is_match_finished = True
                self.end_game_phase = "TIMEOUT"
                self.end_game_timer = 2.0  # Step 1: Announce "Time Out!"
                if self.sim.score_red > self.sim.score_blue:
                    self.victor_name = "Red"
                elif self.sim.score_blue > self.sim.score_red:
                    self.victor_name = "Blue"
                else:
                    self.victor_name = "Draw"

        else:
            # Keep physics running during celebration so players can roam around
            self.sim.step(dt=dt)
            if getattr(self.sim, "kicked_this_step", False):
                self._play_kick_sound()

            self.end_game_timer -= dt

            # Transition between announcement stages
            if self.end_game_timer <= 0.0:
                if self.end_game_phase in ("GOAL_CELEBRATION", "TIMEOUT"):
                    # Step 2: Transition to victorious announcement
                    self.end_game_phase = "VICTORY"
                    self.end_game_timer = 3.0
                    if self.sound_enabled and self.sound_celebration:
                        self.sound_celebration.play()
                elif self.end_game_phase == "VICTORY":
                    # Step 3: Automatically return to menu with persisted config
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    from src.game.states.quick_play_config_state import QuickPlayConfigState
                    self.context.state_manager.change_state(QuickPlayConfigState(self.context))
                    return

        # Smooth Camera Tracking (Follows Keyboard Player, fallback to Red[0])
        human_player = next(
            (p for p, c in zip(self.sim.all_players, self.sim.controllers) if isinstance(c, KeyboardController)),
            self.sim.red_team[0] if self.sim.red_team else None
        )
        track_pos = human_player.pos if human_player else self.sim.center
        cam_target = (track_pos * 0.6) + (self.sim.ball.pos * 0.4)

        p = self.sim.pitch
        world_bounds = pygame.Rect(
            p.outer_left, p.outer_top,
            p.outer_right - p.outer_left, p.outer_bottom - p.outer_top,
        )
        self.camera.update(cam_target, world_bounds, dt)

    def _draw_player_halo(self, surface: pygame.Surface, pos: tuple[int, int], radius: int):
        """Draws the authentic wide translucent ring matching original Haxball."""
        halo_radius = radius + 15
        cache_key = ("halo_wide", halo_radius)

        if cache_key not in self._sprite_cache:
            scale = 4
            pad = 6 * scale
            size = (halo_radius * 2) * scale + pad * 2
            center = size // 2

            hi_res = pygame.Surface((size, size), pygame.SRCALPHA)

            # Soft inner aura
            pygame.draw.circle(
                hi_res,
                (255, 255, 255, 25),
                (center, center),
                halo_radius * scale,
            )
            # Distinct translucent outer ring
            pygame.draw.circle(
                hi_res,
                (255, 255, 255, 120),
                (center, center),
                halo_radius * scale,
                width=int(3.5 * scale),
            )

            final_size = size // scale
            self._sprite_cache[cache_key] = pygame.transform.smoothscale(
                hi_res, (final_size, final_size)
            )

        halo_img = self._sprite_cache[cache_key]
        surface.blit(halo_img, halo_img.get_rect(center=pos))


    def _draw_net(self, surface: pygame.Surface, is_left: bool):
        p = self.sim.pitch
        cam = self.camera

        top_post = cam.apply((p.left if is_left else p.right, p.goal_top))
        bot_post = cam.apply((p.left if is_left else p.right, p.goal_bottom))

        depth = int(p.cfg.GOAL_DEPTH)
        dir_x = -1 if is_left else 1
        curve_offset = 15

        net_top = (top_post[0] + (depth * dir_x), top_post[1] + curve_offset)
        net_bot = (bot_post[0] + (depth * dir_x), bot_post[1] - curve_offset)
        points = [top_post, net_top, net_bot, bot_post]

        min_x = min(pt[0] for pt in points) - 10
        min_y = min(pt[1] for pt in points) - 10
        max_x = max(pt[0] for pt in points) + 10
        max_y = max(pt[1] for pt in points) + 10
        w = max_x - min_x
        h = max_y - min_y

        scale = 4
        scratch = pygame.Surface((w * scale, h * scale), pygame.SRCALPHA)
        scaled_pts = [((pt[0] - min_x) * scale, (pt[1] - min_y) * scale) for pt in points]

        pygame.draw.lines(scratch, (20, 20, 20), False, scaled_pts, 5 * scale)
        for pt in scaled_pts:
            pygame.draw.circle(scratch, (20, 20, 20), pt, int(2.5 * scale))

        smooth_net = pygame.transform.smoothscale(scratch, (w, h))
        surface.blit(smooth_net, (min_x, min_y))

    def draw(self, surface: pygame.Surface):
        # 1. Outer Background
        surface.fill((113, 140, 90))
        p = self.sim.pitch
        cam = self.camera

        # 2. Pitch Geometry & Grass Stripes
        pitch_rect = cam.apply_rect(pygame.Rect(p.left, p.top, p.width, p.height))
        pygame.draw.rect(surface, (119, 153, 91), pitch_rect)

        num_stripes = 12
        stripe_w = p.width / num_stripes
        for i in range(num_stripes):
            if i % 2 == 0:
                s_rect = cam.apply_rect(pygame.Rect(p.left + i * stripe_w, p.top, stripe_w, p.height))
                pygame.draw.rect(surface, (128, 163, 98), s_rect)

        pygame.draw.rect(surface, (240, 240, 240), pitch_rect, width=3)

        # 3. Center Markings
        pygame.draw.line(
            surface, (240, 240, 240),
            cam.apply((p.center.x, p.top)),
            cam.apply((p.center.x, p.bottom)), 3
        )

        c_rad_world = int(p.height * 0.20)
        cx, cy = cam.apply((p.center.x, p.center.y))
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad_world, (240, 240, 240))
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad_world - 1, (240, 240, 240))
        pygame.gfxdraw.filled_circle(surface, cx, cy, 5, (240, 240, 240))

        # 4. Goals & Solid-Color Posts
        self._draw_net(surface, is_left=True)
        self._draw_net(surface, is_left=False)

        for i, post in enumerate(p.posts):
            pos = cam.apply(post.pos)
            post_radius = int(post.radius)
            post_fill = (235, 175, 175) if i < 2 else (175, 195, 235)
            post_surf = self._get_aa_circle(post_radius, post_fill, (20, 20, 20), 3)
            surface.blit(post_surf, post_surf.get_rect(center=pos))

        # 5. Players
        red_slots = [s for s in self.match_config.roster if s.team == "red"]
        for i, player in enumerate(self.sim.red_team):
            pos = cam.apply(player.pos)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)

            # Draw halo if controlled by Keyboard
            slot = red_slots[i] if i < len(red_slots) else None
            if isinstance(getattr(slot, "controller", None), KeyboardController):
                self._draw_player_halo(surface, pos, int(player.radius))

            player_surf = self._get_aa_circle(int(player.radius), (215, 65, 65), out_c, 3)
            surface.blit(player_surf, player_surf.get_rect(center=pos))

            num_txt = self.font_player_num.render(str(i + 1), True, (255, 255, 255))
            surface.blit(num_txt, num_txt.get_rect(center=pos))

        blue_slots = [s for s in self.match_config.roster if s.team == "blue"]
        for j, player in enumerate(self.sim.blue_team):
            pos = cam.apply(player.pos)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)

            # Draw halo if controlled by Keyboard
            slot = blue_slots[j] if j < len(blue_slots) else None
            if isinstance(getattr(slot, "controller", None), KeyboardController):
                self._draw_player_halo(surface, pos, int(player.radius))

            player_surf = self._get_aa_circle(int(player.radius), (70, 115, 220), out_c, 3)
            surface.blit(player_surf, player_surf.get_rect(center=pos))

            num_txt = self.font_player_num.render(str(j + 1), True, (255, 255, 255))
            surface.blit(num_txt, num_txt.get_rect(center=pos))

        # 6. Ball
        ball = self.sim.ball
        pos = cam.apply(ball.pos)
        if self.ball_img:
            img_rect = self.ball_img.get_rect(center=pos)
            surface.blit(self.ball_img, img_rect)
        else:
            ball_surf = self._get_aa_circle(int(ball.radius), (255, 255, 255), (20, 20, 20), 2)
            surface.blit(ball_surf, ball_surf.get_rect(center=pos))

        # 7. Goal / Victory In-Game Announcements
        mode = self.match_config.mode
        if self.is_match_finished:
            self._draw_end_game_announcements(surface)
        elif hasattr(mode, "state") and mode.state == "GOAL_SCORED":
            self._draw_goal_announcement(surface, mode)

        # 8. Segmented HUD Bar
        self._draw_hud(surface)

    def _draw_hud(self, surface: pygame.Surface):
        cx = self.context.screen_width // 2

        # Score Container Pill
        score_w, score_h = 240, 40
        score_rect = pygame.Rect(cx - score_w // 2 - 60, 12, score_w, score_h)
        pygame.draw.rect(surface, (22, 26, 34), score_rect, border_radius=8)

        red_box = pygame.Rect(score_rect.left + 8, score_rect.top + 8, 24, 24)
        pygame.draw.rect(surface, (215, 65, 65), red_box, border_radius=4)

        blue_box = pygame.Rect(score_rect.right - 32, score_rect.top + 8, 24, 24)
        pygame.draw.rect(surface, (70, 115, 220), blue_box, border_radius=4)

        score_str = f"{self.sim.score_red}  -  {self.sim.score_blue}"
        score_txt = self.font_score.render(score_str, True, (245, 245, 245))
        surface.blit(score_txt, score_txt.get_rect(center=score_rect.center))

        # Timer Container Pill
        time_w, time_h = 90, 40
        time_rect = pygame.Rect(score_rect.right + 10, 12, time_w, time_h)
        pygame.draw.rect(surface, (22, 26, 34), time_rect, border_radius=8)

        mode = self.match_config.mode
        if hasattr(mode, "time_remaining"):
            mins = int(mode.time_remaining) // 60
            secs = int(mode.time_remaining) % 60
            timer_surf = self.font_time.render(f"{mins:02d}:{secs:02d}", True, (245, 245, 245))
            surface.blit(timer_surf, timer_surf.get_rect(center=time_rect.center))

        # Sound Button & Menu Button
        mpos = pygame.mouse.get_pos()
        s_hover = self.btn_sound_rect.collidepoint(mpos)
        s_color = (60, 75, 100) if s_hover else (45, 55, 75)
        pygame.draw.rect(surface, s_color, self.btn_sound_rect, border_radius=6)

        icon = self.icon_volume if self.sound_enabled else self.icon_muted
        if icon:
            surface.blit(icon, icon.get_rect(center=self.btn_sound_rect.center))
        else:
            # Fallback text if icon files missing
            lbl = "SND" if self.sound_enabled else "OFF"
            txt = self.font_btn.render(lbl, True, (240, 240, 240))
            surface.blit(txt, txt.get_rect(center=self.btn_sound_rect.center))

        self.btn_pause.draw(surface)

    def _draw_goal_announcement(self, surface: pygame.Surface, mode: Any):
        scorer = self.last_scorer or ("Red" if mode.kickoff_team == "blue" else "Blue")
        team_color = (24, 120, 192) if scorer == "Blue" else (220, 50, 50)

        elapsed = 2.5 - getattr(mode, "state_timer", 2.5)
        alpha = min(255, int((elapsed / 0.4) * 255))

        self._render_banner_text(surface, scorer, "Scores!", team_color, alpha)

    def _draw_end_game_announcements(self, surface: pygame.Surface):
        """Renders the sequential match ending banners with smooth fade-in."""
        if self.end_game_phase == "GOAL_CELEBRATION":
            scorer = self.last_scorer or "Red"
            team_color = (24, 120, 192) if scorer == "Blue" else (220, 50, 50)
            elapsed = 2.2 - self.end_game_timer
            alpha = min(255, int((elapsed / 0.4) * 255))
            self._render_banner_text(surface, scorer, "Scores!", team_color, alpha)

        elif self.end_game_phase == "TIMEOUT":
            elapsed = 2.0 - self.end_game_timer
            alpha = min(255, int((elapsed / 0.4) * 255))
            self._render_banner_text(surface, "Time", "Out!", (240, 240, 240), alpha)

        elif self.end_game_phase == "VICTORY":
            elapsed = 3.0 - self.end_game_timer
            alpha = min(255, int((elapsed / 0.4) * 255))
            if self.victor_name == "Draw":
                self._render_banner_text(surface, "Match", "Drawn!", (235, 235, 235), alpha)
            else:
                color = (24, 120, 192) if self.victor_name == "Blue" else (220, 50, 50)
                self._render_banner_text(surface, self.victor_name, "is Victorious!", color, alpha)

    def _render_banner_text(self, surface: pygame.Surface, line1_str: str, line2_str: str, color, alpha: int):
        line1 = self.font_banner.render(line1_str, True, color)
        line2 = self.font_banner.render(line2_str, True, color)
        sh1 = self.font_banner.render(line1_str, True, (0, 0, 0))
        sh2 = self.font_banner.render(line2_str, True, (0, 0, 0))

        w = max(line1.get_width(), line2.get_width()) + 30
        h = line1.get_height() + line2.get_height() + 20

        banner_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        cx = w // 2
        y1 = 0
        y2 = line1.get_height() - 10

        shadow_offset = 4
        banner_surf.blit(sh1, sh1.get_rect(center=(cx + shadow_offset, y1 + line1.get_height() // 2 + shadow_offset)))
        banner_surf.blit(sh2, sh2.get_rect(center=(cx + shadow_offset, y2 + line2.get_height() // 2 + shadow_offset)))

        banner_surf.blit(line1, line1.get_rect(center=(cx, y1 + line1.get_height() // 2)))
        banner_surf.blit(line2, line2.get_rect(center=(cx, y2 + line2.get_height() // 2)))

        banner_surf.set_alpha(alpha)
        dest_rect = banner_surf.get_rect(center=(self.context.screen_width // 2, self.context.screen_height // 2))
        surface.blit(banner_surf, dest_rect)