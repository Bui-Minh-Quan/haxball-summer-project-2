import pygame
from src.engine.vector import Vec2

class Camera:
    def __init__(self, viewport_w: int, viewport_h: int):
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.offset = Vec2(0, 0)

    def update(self, target: Vec2, bounds_rect: pygame.Rect):
        # 1. Center the camera on the target
        self.offset.x = target.x - (self.viewport_w / 2)
        self.offset.y = target.y - (self.viewport_h / 2)

        # 2. Clamp Camera to World Bounds
        # If the window is smaller than the pitch, prevent scrolling past the outer fences
        if self.viewport_w < bounds_rect.width:
            self.offset.x = max(bounds_rect.left, min(self.offset.x, bounds_rect.right - self.viewport_w))
        else:
            # If window is larger than pitch, center the pitch on screen
            self.offset.x = bounds_rect.left - (self.viewport_w - bounds_rect.width) / 2

        if self.viewport_h < bounds_rect.height:
            self.offset.y = max(bounds_rect.top, min(self.offset.y, bounds_rect.bottom - self.viewport_h))
        else:
            self.offset.y = bounds_rect.top - (self.viewport_h - bounds_rect.height) / 2

    def apply(self, pos: Vec2 | tuple[float, float]) -> tuple[int, int]:
        """Convert World coordinates to Screen coordinates."""
        if isinstance(pos, Vec2):
            return (int(pos.x - self.offset.x), int(pos.y - self.offset.y))
        return (int(pos[0] - self.offset.x), int(pos[1] - self.offset.y))

    def apply_rect(self, rect: pygame.Rect) -> pygame.Rect:
        """Offset a Pygame Rect for rendering."""
        return pygame.Rect(
            rect.left - self.offset.x,
            rect.top - self.offset.y,
            rect.width,
            rect.height
        )