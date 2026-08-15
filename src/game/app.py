import sys
import pygame
from src.game.state_manager import GameContext, StateManager
from src.game.states.menu_state import MenuState


class App:

    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        pygame.font.init()
        self.context = GameContext(screen_width=width, screen_height=height)
        self.manager = StateManager(self.context)
        self.manager.change_state(MenuState(self.context))

    def run(self):
        while self.context.running:
            dt = min(self.context.clock.tick(60) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.context.running = False
                else:
                    self.manager.handle_event(event)

            if not self.context.running:
                break

            self.manager.update(dt)
            self.manager.draw(self.context.screen)
            pygame.display.flip()

        pygame.quit()
        sys.exit()