import pygame
from src.game.state_manager import GameState
from src.game.ui.button import Button


class PauseState(GameState):

    def __init__(self, context):
        super().__init__(context)
        self.font_title = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 20, bold=True)

        # Semi-transparent overlay
        self.overlay = pygame.Surface(
            (context.screen_width, context.screen_height), pygame.SRCALPHA
        )
        self.overlay.fill((10, 12, 18, 180))

        cx = context.screen_width // 2
        self.buttons = [
            Button(
                pygame.Rect(cx - 110, 310, 220, 45),
                "Continue",
                self.font_btn,
                self._resume,
            ),
            Button(
                pygame.Rect(cx - 110, 370, 220, 45),
                "Main Menu",
                self.font_btn,
                self._to_menu,
            ),
            Button(
                pygame.Rect(cx - 110, 430, 220, 45),
                "Exit",
                self.font_btn,
                self._quit,
                base_color=(70, 40, 45),
                hover_color=(200, 60, 60),
            ),
        ]

    def _resume(self):
        self.context.state_manager.pop_state()

    def _to_menu(self):
        from src.game.states.menu_state import MenuState

        self.context.state_manager.change_state(MenuState(self.context))

    def _quit(self):
        self.context.running = False

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and (
            event.key == pygame.K_ESCAPE or event.key == pygame.K_p
        ):
            self._resume()
            return
        for btn in self.buttons:
            btn.handle_event(event)

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        surface.blit(self.overlay, (0, 0))

        # Modal Frame
        modal_rect = pygame.Rect(0, 0, 320, 340)
        modal_rect.center = (
            self.context.screen_width // 2,
            self.context.screen_height // 2 + 20,
        )
        pygame.draw.rect(surface, (30, 34, 46), modal_rect, border_radius=10)
        pygame.draw.rect(
            surface, (90, 100, 125), modal_rect, width=2, border_radius=10
        )

        title = self.font_title.render("PAUSED", True, (255, 255, 255))
        surface.blit(
            title,
            title.get_rect(center=(self.context.screen_width // 2, 240)),
        )

        for btn in self.buttons:
            btn.draw(surface)