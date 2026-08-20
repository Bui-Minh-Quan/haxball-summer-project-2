import pygame
from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import HeuristicBot
from src.engine.controllers import HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.game.controllers import KeyboardController
from src.game.state_manager import GameState
from src.game.states.play_state import PlayState
from src.game.ui.button import Button


class MenuState(GameState):

    def __init__(self, context):
        super().__init__(context)
        self.font_title = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 22, bold=True)

        center_x = context.screen_width // 2
        self.buttons = [
            Button(
                pygame.Rect(center_x - 120, 360, 240, 50),
                "1v1 vs Bot",
                self.font_btn,
                self._start_game,
            ),
            Button(
                pygame.Rect(center_x - 120, 430, 240, 50),
                "Exit Game",
                self.font_btn,
                self._quit_game,
                base_color=(70, 40, 45),
                hover_color=(200, 60, 60),
            ),
        ]

    def _start_game(self):
        # Uses defaults defined in MatchConfig without hardcoded overrides
        match_cfg = MatchConfig(
            mode=ClassicMatchMode(),
            roster=[
                PlayerSlot(
                    team="red",
                    stats=PlayerStats(name="Human Player", accel=3200.0),
                    controller=KeyboardController(),
                ),
                PlayerSlot(
                    team="blue",
                    stats=PlayerStats(name="Bot Blue", accel=3000.0),
                    controller=HeuristicBotController(
                        HeuristicBot(team="blue")
                    ),
                ),
            ],
        )
        self.context.state_manager.change_state(
            PlayState(self.context, match_config=match_cfg)
        )

    def _quit_game(self):
        self.context.running = False

    def handle_event(self, event: pygame.event.Event):
        for btn in self.buttons:
            btn.handle_event(event)

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        surface.fill((22, 25, 34))

        title_surf = self.font_title.render(
            "HAXBALL AI", True, (255, 255, 255)
        )
        sub_surf = self.font_btn.render(
            "Reinforcement Learning Edition", True, (100, 160, 255)
        )
        surface.blit(
            title_surf,
            title_surf.get_rect(
                center=(self.context.screen_width // 2, 220)
            ),
        )
        surface.blit(
            sub_surf,
            sub_surf.get_rect(
                center=(self.context.screen_width // 2, 275)
            ),
        )

        for btn in self.buttons:
            btn.draw(surface)