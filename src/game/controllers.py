import pygame
from src.engine.controllers import Controller
from src.engine.vector import Vec2
from typing import Any


class KeyboardController(Controller):
    """Polls Pygame keyboard state for a player slot."""

    def __init__(
        self,
        up=pygame.K_w,
        down=pygame.K_s,
        left=pygame.K_a,
        right=pygame.K_d,
        kick=pygame.K_SPACE,
    ):
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.kick = kick

    def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
        keys = pygame.key.get_pressed()
        move = Vec2(0.0, 0.0)
        if keys[self.up]:
            move.y -= 1.0
        if keys[self.down]:
            move.y += 1.0
        if keys[self.left]:
            move.x -= 1.0
        if keys[self.right]:
            move.x += 1.0
        is_kicking = keys[self.kick]
        return move, is_kicking