import pygame


class Overlay():
    def __init__(self, surface, rect,
                 active=True,
                 on_focus=True,
                 selectable=True,
                 bg_color=(45, 45, 45),
                 border_color=(100, 100, 100),
                 focus_color=(200, 200, 60),
                 custom_draw=None,
                 custom_callback=None,
                 ):
        self.surface = surface
        self.rect = rect

        self.bg_color = bg_color
        self.border_color = border_color
        self.focus_color = focus_color

        self.active = active
        self.on_focus = on_focus
        self.selectable = selectable

        self.custom_draw = custom_draw
        self.custom_callback = custom_callback

    def collision(self, point: tuple[int, int]) -> bool:
        if not self.selectable:
            return False
        xmin = self.rect[0]
        xmax = self.rect[0] + self.rect[2]
        ymin = self.rect[1]
        ymax = self.rect[1] + self.rect[3]
        return (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax)

    def draw(self):
        if self.active:
            r_border = 15
            pygame.draw.rect(self.surface, self.bg_color, self.rect, border_radius=r_border)
            pygame.draw.rect(self.surface, self.focus_color if self.on_focus else self.border_color, self.rect, border_radius=r_border, width=2)

            if self.custom_draw is not None:
                self.custom_draw(self.surface, rect=self.rect)

    def callback(self):
        if self.custom_callback is not None:
            self.custom_callback()
