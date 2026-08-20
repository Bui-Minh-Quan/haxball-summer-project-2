import pygame
from src.game.state_manager import GameState
from src.game.states.quick_play_config_state import QuickPlayConfigState
from src.game.ui.button import Button


class MenuState(GameState):

    def __init__(self, context):
        super().__init__(context)
        self.font_title = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 22, bold=True)

        cx = context.screen_width // 2
        self.buttons = [
            Button(
                pygame.Rect(cx - 130, 340, 260, 52),
                "Quick Play",
                self.font_btn,
                self._open_quick_play_config,
                base_color=(35, 115, 60),
                hover_color=(45, 145, 75),
            ),
            Button(
                pygame.Rect(cx - 130, 410, 260, 52),
                "Exit Game",
                self.font_btn,
                self._quit_game,
                base_color=(70, 40, 45),
                hover_color=(190, 55, 60),
            ),
        ]

    def _open_quick_play_config(self):
        self.context.state_manager.change_state(QuickPlayConfigState(self.context))

    def _quit_game(self):
        self.context.running = False

    def handle_event(self, event: pygame.event.Event):
        for btn in self.buttons:
            btn.handle_event(event)

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        surface.fill((20, 23, 31))
        cx = self.context.screen_width // 2

        title_surf = self.font_title.render("HAXBALL AI", True, (255, 255, 255))
        sub_surf = self.font_sub.render("Tactical Multi-Agent Engine", True, (100, 160, 255))

        surface.blit(title_surf, title_surf.get_rect(center=(cx, 210)))
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, 265)))

        for btn in self.buttons:
            btn.draw(surface)