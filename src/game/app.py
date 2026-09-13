import sys
import pygame
from src.game.state_manager import GameContext, StateManager
from src.game.states.menu_state import MenuState


class App:
    """Core game application utilizing a hardware-scaled Virtual Canvas."""

    VIRTUAL_WIDTH = 1600
    VIRTUAL_HEIGHT = 900

    def __init__(self, width: int | None = None, height: int | None = None):
        pygame.init()
        pygame.font.init()

        # Shared context using the virtual resolution as reference coordinate space
        self.context = GameContext(
            screen_width=self.VIRTUAL_WIDTH,
            screen_height=self.VIRTUAL_HEIGHT,
        )

        # Hardware-scaled logical viewport
        self.context.screen = pygame.display.set_mode(
            (self.VIRTUAL_WIDTH, self.VIRTUAL_HEIGHT),
            pygame.SCALED | pygame.RESIZABLE,
        )
        pygame.display.set_caption("Haxball AI")

        self.manager = StateManager(self.context)
        self.manager.change_state(MenuState(self.context))

    def run(self):
        while self.context.running:
            dt = min(self.context.clock.tick(60) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.context.running = False

                # Native Fullscreen Toggle
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()

                else:
                    self.manager.handle_event(event)

            if not self.context.running:
                break

            self.manager.update(dt)
            self.manager.draw(self.context.screen)
            pygame.display.flip()

        pygame.quit()
        sys.exit()