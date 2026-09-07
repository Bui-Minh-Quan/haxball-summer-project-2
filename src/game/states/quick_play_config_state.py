import os
from pathlib import Path
import pygame
from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import HeuristicBotController, NumpyRLController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.game.controllers import KeyboardController
from src.game.state_manager import GameState
from src.game.states.play_state import PlayState
from src.game.ui.button import Button
from src.rl.numpy_actor import NumpyActor


_LOADED_ACTORS: dict[str, NumpyActor] = {}

class Dropup:
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        options: list[str],
        values: list,
        default_idx: int = 0,
        max_visible: int = 6,
    ):
        self.rect = rect
        self.label = label
        self.options = options
        self.values = values
        self.selected_idx = default_idx
        self.max_visible = max_visible
        self.is_open = False
        self.scroll_offset = 0
        self.item_h = 32

    @property
    def selected_value(self):
        return self.values[self.selected_idx]

    @property
    def selected_text(self):
        return self.options[self.selected_idx]

    def get_menu_rect(self) -> pygame.Rect:
        visible_count = min(len(self.options), self.max_visible)
        menu_h = visible_count * self.item_h
        menu_y = self.rect.y - menu_h - 4
        if menu_y < 60:
            menu_y = self.rect.bottom + 4
        return pygame.Rect(self.rect.x, menu_y, self.rect.width, menu_h)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mpos = event.pos
            if self.is_open:
                menu_rect = self.get_menu_rect()
                if menu_rect.collidepoint(mpos):
                    rel_y = mpos[1] - menu_rect.y
                    clicked_slot = rel_y // self.item_h
                    target_idx = self.scroll_offset + clicked_slot
                    if 0 <= target_idx < len(self.options):
                        self.selected_idx = target_idx
                    self.is_open = False
                    return True
                elif self.rect.collidepoint(mpos):
                    self.is_open = False
                    return True
                else:
                    self.is_open = False
                    return False
            else:
                if self.rect.collidepoint(mpos):
                    self.is_open = True
                    max_scroll = max(0, len(self.options) - self.max_visible)
                    self.scroll_offset = max(
                        0,
                        min(max_scroll, self.selected_idx - self.max_visible // 2),
                    )
                    return True

        elif event.type == pygame.MOUSEWHEEL and self.is_open:
            menu_rect = self.get_menu_rect()
            if menu_rect.collidepoint(pygame.mouse.get_pos()):
                max_scroll = max(0, len(self.options) - self.max_visible)
                self.scroll_offset = max(
                    0, min(max_scroll, self.scroll_offset - event.y)
                )
                return True

        return False

    def draw_button(
        self, surface: pygame.Surface, font_btn: pygame.font.Font, font_lbl: pygame.font.Font
    ):
        lbl_surf = font_lbl.render(self.label, True, (135, 145, 170))
        surface.blit(lbl_surf, (self.rect.x, self.rect.y - 20))

        mpos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mpos)
        base_color = (40, 48, 66) if is_hover or self.is_open else (28, 34, 48)
        border_color = (90, 120, 175) if self.is_open else ((60, 72, 98) if is_hover else (45, 54, 75))

        pygame.draw.rect(surface, base_color, self.rect, border_radius=8)
        pygame.draw.rect(surface, border_color, self.rect, width=2, border_radius=8)

        val_surf = font_btn.render(self.selected_text, True, (245, 245, 250))
        val_rect = val_surf.get_rect(midleft=(self.rect.x + 14, self.rect.centery))
        surface.blit(val_surf, val_rect)

        caret_char = "▲" if self.is_open else "▼"
        caret_surf = font_btn.render(caret_char, True, (110, 125, 155))
        caret_rect = caret_surf.get_rect(midright=(self.rect.right - 14, self.rect.centery))
        surface.blit(caret_surf, caret_rect)

    def draw_menu(self, surface: pygame.Surface, font_btn: pygame.font.Font):
        if not self.is_open:
            return

        menu_rect = self.get_menu_rect()
        pygame.draw.rect(surface, (20, 24, 34), menu_rect, border_radius=8)
        pygame.draw.rect(surface, (70, 95, 145), menu_rect, width=2, border_radius=8)

        mpos = pygame.mouse.get_pos()
        visible_count = min(len(self.options), self.max_visible)

        for slot in range(visible_count):
            opt_idx = self.scroll_offset + slot
            if opt_idx >= len(self.options):
                break

            slot_rect = pygame.Rect(
                menu_rect.x + 2,
                menu_rect.y + slot * self.item_h,
                menu_rect.width - 4,
                self.item_h,
            )

            is_selected = opt_idx == self.selected_idx
            is_hover = slot_rect.collidepoint(mpos)

            if is_selected:
                pygame.draw.rect(surface, (35, 75, 130), slot_rect, border_radius=6)
            elif is_hover:
                pygame.draw.rect(surface, (36, 44, 62), slot_rect, border_radius=6)

            text_color = (255, 255, 255) if (is_selected or is_hover) else (180, 190, 210)
            item_surf = font_btn.render(self.options[opt_idx], True, text_color)
            item_rect = item_surf.get_rect(midleft=(slot_rect.x + 12, slot_rect.centery))
            surface.blit(item_surf, item_rect)

        if len(self.options) > self.max_visible:
            sb_x = menu_rect.right - 8
            sb_h = menu_rect.height
            ratio = self.max_visible / len(self.options)
            thumb_h = max(14, int(sb_h * ratio))
            max_offset = len(self.options) - self.max_visible
            thumb_y = menu_rect.y + int((self.scroll_offset / max_offset) * (sb_h - thumb_h))

            thumb_rect = pygame.Rect(sb_x, thumb_y, 4, thumb_h)
            pygame.draw.rect(surface, (90, 110, 150), thumb_rect, border_radius=2)


class QuickPlayConfigState(GameState):

    def __init__(self, context):
        super().__init__(context)
        self.font_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 16)
        self.font_lbl = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_action = pygame.font.SysFont("Arial", 18, bold=True)

        self.stadium_dimensions = {
            "Classic": (1200.0, 800.0),
            "Small": (960.0, 640.0),
            "Big": (1440.0, 960.0),
            "Huge": (1680.0, 1120.0),
        }

        # Load persisted settings if returning from a match
        self.saved_cfg = getattr(self.context, "last_quick_play_config", None)

        self._init_dropups()
        self._init_action_buttons()

    def _init_dropups(self):
        cx = self.context.screen_width // 2
        col_w = 340
        col_gap = 50
        left_x = cx - col_w - (col_gap // 2)
        right_x = cx + (col_gap // 2)

        s = self.saved_cfg or {}

        # 1. Team
        self.drop_team = Dropup(
            rect=pygame.Rect(left_x, 240, col_w, 42),
            label="PLAYER TEAM",
            options=["Red", "Blue"],
            values=["red", "blue"],
            default_idx=s.get("team_idx", 0),
        )

        # 2. Opponent
        self.drop_opponent = Dropup(
            rect=pygame.Rect(left_x, 340, col_w, 42),
            label="OPPONENT TYPE",
            options=[
                "Heuristic",
                "Easy RL Agent (Stage 2)",
                "Medium RL Agent (Stage 3)",
            ],
            values=["heuristic", "rl_stage2", "rl_stage3"],
            default_idx=s.get("opp_idx", 0),
        )

        # 3. Stadium
        self.drop_stadium = Dropup(
            rect=pygame.Rect(left_x, 440, col_w, 42),
            label="STADIUM PRESET",
            options=["Classic", "Small", "Big", "Huge"],
            values=["Classic", "Small", "Big", "Huge"],
            default_idx=s.get("stadium_idx", 0),
        )

        # 4. Time Limit
        time_options = ["Infinite"] + [f"{m} min" for m in range(1, 16)]
        time_values = [0.0] + [float(m * 60) for m in range(1, 16)]
        self.drop_time = Dropup(
            rect=pygame.Rect(right_x, 240, col_w, 42),
            label="TIME LIMIT",
            options=time_options,
            values=time_values,
            default_idx=s.get("time_idx", 3),  # 3 min default
            max_visible=7,
        )

        # 5. Score Limit
        score_options = ["Infinite"] + [f"{g} Goal{'s' if g > 1 else ''}" for g in range(1, 16)]
        score_values = list(range(0, 16))
        self.drop_score = Dropup(
            rect=pygame.Rect(right_x, 340, col_w, 42),
            label="SCORE LIMIT",
            options=score_options,
            values=score_values,
            default_idx=s.get("score_idx", 3),  # 3 goals default
            max_visible=7,
        )

        self.dropups = [
            self.drop_team,
            self.drop_opponent,
            self.drop_stadium,
            self.drop_time,
            self.drop_score,
        ]

    def _init_action_buttons(self):
        cx = self.context.screen_width // 2
        btn_y = 560
        btn_w, btn_h, gap = 175, 48, 20

        self.btn_start = Button(
            pygame.Rect(cx - btn_w - (gap // 2), btn_y, btn_w, btn_h),
            "Start Match",
            self.font_action,
            self._start_match,
            base_color=(35, 130, 65),
            hover_color=(45, 165, 80),
        )

        self.btn_back = Button(
            pygame.Rect(cx + (gap // 2), btn_y, btn_w, btn_h),
            "Back to Menu",
            self.font_action,
            self._go_back,
            base_color=(50, 56, 72),
            hover_color=(70, 78, 98),
        )

    def _get_or_load_actor(self, npz_path: str) -> NumpyActor | None:
        """Loads NumpyActor into RAM and caches it for subsequent matches."""
        if npz_path in _LOADED_ACTORS:
            return _LOADED_ACTORS[npz_path]

        if os.path.exists(npz_path):
            try:
                actor = NumpyActor(npz_path)
                _LOADED_ACTORS[npz_path] = actor
                return actor
            except Exception as e:
                print(f"[Warning] Failed to load model from {npz_path}: {e}")
        return None

    def _resolve_opponent_controller(self, opp_type: str, opp_team: str):
        """Resolves opponent controller using pure NumPy inference without PyTorch."""
        if opp_type == "heuristic":
            coord = TeamHeuristicCoordinator(team=opp_team)
            return HeuristicBotController(coord), "Heuristic Bot"

        # Map UI choice to the exported .npz model
        model_filename = (
            "stage2_actor.npz" if opp_type == "rl_stage2" else "stage3_actor.npz"
        )
        display_label = (
            "Easy RL" if opp_type == "rl_stage2" else "Medium RL"
        )
        npz_path = os.path.join("assets", "models", model_filename)

        actor = self._get_or_load_actor(npz_path)
        if actor is not None:
            controller = NumpyRLController(actor, team=opp_team)
            return controller, display_label

        # Fallback to Heuristic Bot if .npz file is missing
        print(f"[Warning] Model {npz_path} not found. Falling back to Heuristic Bot.")
        coord = TeamHeuristicCoordinator(team=opp_team)
        return HeuristicBotController(coord), "Heuristic Bot"

    def _start_match(self):
        # Save indices so returning to this menu remembers everything
        self.context.last_quick_play_config = {
            "team_idx": self.drop_team.selected_idx,
            "opp_idx": self.drop_opponent.selected_idx,
            "stadium_idx": self.drop_stadium.selected_idx,
            "time_idx": self.drop_time.selected_idx,
            "score_idx": self.drop_score.selected_idx,
        }

        user_team = self.drop_team.selected_value
        opp_team = "blue" if user_team == "red" else "red"

        opp_type = self.drop_opponent.selected_value
        opp_ctrl, opp_name = self._resolve_opponent_controller(opp_type, opp_team)

        # Do NOT hardcode accel! Uses whatever you configured in PlayerStats default
        user_slot = PlayerSlot(
            team=user_team,
            stats=PlayerStats(name="Player"),
            controller=KeyboardController(),
        )
        opp_slot = PlayerSlot(
            team=opp_team,
            stats=PlayerStats(name=opp_name),
            controller=opp_ctrl,
        )

        roster = [user_slot, opp_slot] if user_team == "red" else [opp_slot, user_slot]

        stadium_name = self.drop_stadium.selected_value
        pitch_w, pitch_h = self.stadium_dimensions[stadium_name]

        time_lim = self.drop_time.selected_value
        score_lim = self.drop_score.selected_value

        match_cfg = MatchConfig(
            mode=ClassicMatchMode(time_limit=time_lim, score_limit=score_lim),
            roster=roster,
            pitch_width=pitch_w,
            pitch_height=pitch_h,
            time_limit=time_lim,
            score_limit=score_lim,
        )

        self.context.state_manager.change_state(
            PlayState(self.context, match_config=match_cfg)
        )

    def _go_back(self):
        from src.game.states.menu_state import MenuState
        self.context.state_manager.change_state(MenuState(self.context))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._go_back()
            return

        active_drop = next((d for d in self.dropups if d.is_open), None)
        if active_drop:
            handled = active_drop.handle_event(event)
            if handled:
                return

        for drop in self.dropups:
            if drop.handle_event(event):
                for other in self.dropups:
                    if other != drop:
                        other.is_open = False
                return

        self.btn_start.handle_event(event)
        self.btn_back.handle_event(event)

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        surface.fill((16, 19, 26))
        cx = self.context.screen_width // 2

        t_surf = self.font_title.render("1v1 MATCH SETUP", True, (245, 245, 250))
        surface.blit(t_surf, t_surf.get_rect(center=(cx, 85)))

        sub_surf = self.font_sub.render(
            "Select match rules, arena scale, and opponent configuration",
            True,
            (115, 128, 155),
        )
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, 125)))

        panel_rect = pygame.Rect(cx - 390, 175, 780, 345)
        pygame.draw.rect(surface, (22, 26, 36), panel_rect, border_radius=12)
        pygame.draw.rect(surface, (38, 45, 62), panel_rect, width=2, border_radius=12)

        for drop in self.dropups:
            drop.draw_button(surface, self.font_btn, self.font_lbl)

        self.btn_start.draw(surface)
        self.btn_back.draw(surface)

        active_drop = next((d for d in self.dropups if d.is_open), None)
        if active_drop:
            active_drop.draw_menu(surface, self.font_btn)