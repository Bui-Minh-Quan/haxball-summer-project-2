import pygame
from src.engine.vector import Vec2


class Camera:

    def __init__(
        self, viewport_w: int, viewport_h: int, hud_height: int = 50
    ):
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.hud_height = hud_height
        # Usable screen height below the top HUD bar
        self.usable_h = viewport_h - hud_height

        self.offset = Vec2(0, 0)
        self.smooth_speed = 7.0

    def update(self, target: Vec2, bounds_rect: pygame.Rect, dt: float):
        # 1. Target centering in the usable gameplay area
        desired_x = target.x - (self.viewport_w / 2)
        desired_y = target.y - (self.usable_h / 2)

        # 2. Horizontal Clamping
        if self.viewport_w < bounds_rect.width:
            desired_x = max(
                bounds_rect.left, min(desired_x, bounds_rect.right - self.viewport_w)
            )
        else:
            desired_x = bounds_rect.left - (self.viewport_w - bounds_rect.width) / 2

        # 3. Vertical Clamping (Respects HUD height)
        if self.usable_h < bounds_rect.height:
            desired_y = max(
                bounds_rect.top, min(desired_y, bounds_rect.bottom - self.usable_h)
            )
        else:
            desired_y = bounds_rect.top - (self.usable_h - bounds_rect.height) / 2

        # 4. Smooth Lerp
        self.offset.x += (desired_x - self.offset.x) * self.smooth_speed * dt
        self.offset.y += (desired_y - self.offset.y) * self.smooth_speed * dt

    def apply(self, pos: Vec2 | tuple[float, float]) -> tuple[int, int]:
        """Maps World Space -> Screen Space (shifted below HUD)."""
        px = pos.x if isinstance(pos, Vec2) else pos[0]
        py = pos.y if isinstance(pos, Vec2) else pos[1]
        return (
            int(px - self.offset.x),
            int(py - self.offset.y + self.hud_height),
        )

    def apply_rect(self, rect: pygame.Rect) -> pygame.Rect:
        """Translates a Pygame Rect below the HUD."""
        return pygame.Rect(
            rect.left - self.offset.x,
            rect.top - self.offset.y + self.hud_height,
            rect.width,
            rect.height,
        )