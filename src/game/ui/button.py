import pygame


class Button:

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        callback,
        base_color=(50, 55, 70),
        hover_color=(75, 130, 230),
        text_color=(255, 255, 255),
    ):
        self.rect = rect
        self.text = text
        self.font = font
        self.callback = callback
        self.base_color = base_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

        self.text_surf = self.font.render(self.text, True, self.text_color)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                self.callback()

    def draw(self, surface: pygame.Surface):
        color = self.hover_color if self.is_hovered else self.base_color
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(
            surface, (200, 210, 230), self.rect, width=2, border_radius=8
        )
        surface.blit(self.text_surf, self.text_rect)