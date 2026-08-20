import pygame
from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.game.controllers import KeyboardController
from src.game.state_manager import GameState
from src.game.states.play_state import PlayState
from src.game.ui.button import Button


class QuickPlayConfigState(GameState):

    def __init__(self, context):
        super().__init__(context)
        self.font_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_section = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_val = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_action = pygame.font.SysFont("Arial", 18, bold=True)

        # Match Defaults
        self.red_count = 1
        self.blue_count = 1
        self.time_limits = [60.0, 120.0, 180.0, 300.0, 0.0]  # 0.0 = Unlimited
        self.time_idx = 2  # 180s (3 min)
        self.score_limits = [1, 3, 5, 7, 0]  # 0 = Unlimited
        self.score_idx = 1  # 3 goals

        self._build_ui()

    def _build_ui(self):
        cx = self.context.screen_width // 2
        self.buttons: list[Button] = []

        # 1. Preset Buttons (1v1 through 5v5)
        preset_w, preset_h, gap = 64, 34, 10
        total_p_w = (preset_w * 5) + (gap * 4)
        p_start_x = cx - (total_p_w // 2)
        p_y = 145

        for i, count in enumerate([1, 2, 3, 4, 5]):
            btn = Button(
                pygame.Rect(p_start_x + i * (preset_w + gap), p_y, preset_w, preset_h),
                f"{count}v{count}",
                self.font_btn,
                lambda c=count: self._set_preset(c),
            )
            self.buttons.append(btn)

        # Layout Column Anchors (Left = cx - 140, Right = cx + 140)
        col_left = cx - 140
        col_right = cx + 140
        btn_s = 34
        half_span = 70  # Distance from column center to stepper buttons

        # 2. Team Size Steppers
        stepper_y = 250
        # Red Stepper
        self.buttons.append(Button(
            pygame.Rect(col_left - half_span - btn_s // 2, stepper_y, btn_s, btn_s),
            "-", self.font_btn, lambda: self._adj_red(-1),
            base_color=(160, 45, 45), hover_color=(200, 60, 60)
        ))
        self.buttons.append(Button(
            pygame.Rect(col_left + half_span - btn_s // 2, stepper_y, btn_s, btn_s),
            "+", self.font_btn, lambda: self._adj_red(1),
            base_color=(160, 45, 45), hover_color=(200, 60, 60)
        ))

        # Blue Stepper
        self.buttons.append(Button(
            pygame.Rect(col_right - half_span - btn_s // 2, stepper_y, btn_s, btn_s),
            "-", self.font_btn, lambda: self._adj_blue(-1),
            base_color=(40, 85, 170), hover_color=(55, 110, 215)
        ))
        self.buttons.append(Button(
            pygame.Rect(col_right + half_span - btn_s // 2, stepper_y, btn_s, btn_s),
            "+", self.font_btn, lambda: self._adj_blue(1),
            base_color=(40, 85, 170), hover_color=(55, 110, 215)
        ))

        # 3. Match Rules Steppers (Time & Score Limits)
        rules_y = 365
        # Time Stepper
        self.buttons.append(Button(
            pygame.Rect(col_left - half_span - btn_s // 2, rules_y, btn_s, btn_s),
            "<", self.font_btn, lambda: self._adj_time(-1)
        ))
        self.buttons.append(Button(
            pygame.Rect(col_left + half_span - btn_s // 2, rules_y, btn_s, btn_s),
            ">", self.font_btn, lambda: self._adj_time(1)
        ))

        # Score Stepper
        self.buttons.append(Button(
            pygame.Rect(col_right - half_span - btn_s // 2, rules_y, btn_s, btn_s),
            "<", self.font_btn, lambda: self._adj_score(-1)
        ))
        self.buttons.append(Button(
            pygame.Rect(col_right + half_span - btn_s // 2, rules_y, btn_s, btn_s),
            ">", self.font_btn, lambda: self._adj_score(1)
        ))

        # 4. Action Buttons (Start Match & Back)
        action_y = 470
        self.buttons.append(Button(
            pygame.Rect(cx - 180, action_y, 165, 48),
            "Start Match", self.font_action, self._start_match,
            base_color=(35, 125, 60), hover_color=(45, 160, 75)
        ))
        self.buttons.append(Button(
            pygame.Rect(cx + 15, action_y, 165, 48),
            "Back to Menu", self.font_action, self._go_back,
            base_color=(55, 60, 75), hover_color=(75, 82, 100)
        ))

    def _set_preset(self, count: int):
        self.red_count = count
        self.blue_count = count

    def _adj_red(self, delta: int):
        self.red_count = max(1, min(5, self.red_count + delta))

    def _adj_blue(self, delta: int):
        self.blue_count = max(1, min(5, self.blue_count + delta))

    def _adj_time(self, delta: int):
        self.time_idx = (self.time_idx + delta) % len(self.time_limits)

    def _adj_score(self, delta: int):
        self.score_idx = (self.score_idx + delta) % len(self.score_limits)

    def _start_match(self):
        red_coord = TeamHeuristicCoordinator(team="red")
        blue_coord = TeamHeuristicCoordinator(team="blue")
        roster = []

        # Red Team: 1 Keyboard Controller + (N - 1) Bots
        roster.append(PlayerSlot(
            team="red",
            stats=PlayerStats(name="Player", accel=3200.0),
            controller=KeyboardController(),
        ))
        for i in range(1, self.red_count):
            roster.append(PlayerSlot(
                team="red",
                stats=PlayerStats(name=f"Red {i + 1}", accel=3000.0),
                controller=HeuristicBotController(red_coord),
            ))

        # Blue Team: M Bots
        for i in range(self.blue_count):
            roster.append(PlayerSlot(
                team="blue",
                stats=PlayerStats(name=f"Blue {i + 1}", accel=3000.0),
                controller=HeuristicBotController(blue_coord),
            ))

        time_lim = self.time_limits[self.time_idx]
        score_lim = self.score_limits[self.score_idx]

        match_cfg = MatchConfig(
            mode=ClassicMatchMode(time_limit=time_lim, score_limit=score_lim),
            roster=roster,
            time_limit=time_lim,
            score_limit=score_lim,
        )
        self.context.state_manager.change_state(PlayState(self.context, match_config=match_cfg))

    def _go_back(self):
        from src.game.states.menu_state import MenuState
        self.context.state_manager.change_state(MenuState(self.context))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._go_back()
        for btn in self.buttons:
            btn.handle_event(event)

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        surface.fill((20, 23, 31))
        cx = self.context.screen_width // 2
        col_left = cx - 140
        col_right = cx + 140

        # Title
        t_surf = self.font_title.render("MATCH CONFIGURATION", True, (245, 245, 245))
        surface.blit(t_surf, t_surf.get_rect(center=(cx, 55)))

        # Section 1: Presets
        s1 = self.font_section.render("QUICK PRESETS", True, (120, 130, 150))
        surface.blit(s1, s1.get_rect(center=(cx, 118)))

        # Section 2: Team Sizes
        s2 = self.font_section.render("TEAM COMPOSITION", True, (120, 130, 150))
        surface.blit(s2, s2.get_rect(center=(cx, 205)))

        # Red & Blue Values
        r_lbl = self.font_section.render("RED TEAM", True, (230, 80, 80))
        surface.blit(r_lbl, r_lbl.get_rect(center=(col_left, 230)))
        r_val = self.font_val.render(f"{self.red_count} Players", True, (240, 240, 240))
        surface.blit(r_val, r_val.get_rect(center=(col_left, 267)))

        b_lbl = self.font_section.render("BLUE TEAM", True, (80, 140, 240))
        surface.blit(b_lbl, b_lbl.get_rect(center=(col_right, 230)))
        b_val = self.font_val.render(f"{self.blue_count} Players", True, (240, 240, 240))
        surface.blit(b_val, b_val.get_rect(center=(col_right, 267)))

        # Section 3: Rules
        s3 = self.font_section.render("MATCH RULES", True, (120, 130, 150))
        surface.blit(s3, s3.get_rect(center=(cx, 320)))

        # Time & Score Values
        t_lbl = self.font_section.render("TIME LIMIT", True, (160, 170, 190))
        surface.blit(t_lbl, t_lbl.get_rect(center=(col_left, 345)))
        t_raw = self.time_limits[self.time_idx]
        t_str = "No Limit" if t_raw == 0.0 else f"{int(t_raw // 60)} min"
        t_val = self.font_val.render(t_str, True, (240, 240, 240))
        surface.blit(t_val, t_val.get_rect(center=(col_left, 382)))

        s_lbl = self.font_section.render("SCORE LIMIT", True, (160, 170, 190))
        surface.blit(s_lbl, s_lbl.get_rect(center=(col_right, 345)))
        s_raw = self.score_limits[self.score_idx]
        s_str = "No Limit" if s_raw == 0 else f"{s_raw} Goals"
        s_val = self.font_val.render(s_str, True, (240, 240, 240))
        surface.blit(s_val, s_val.get_rect(center=(col_right, 382)))

        # Draw Buttons
        for btn in self.buttons:
            btn.draw(surface)