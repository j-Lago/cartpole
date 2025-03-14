import pygame
import pygame.gfxdraw
import math
from pendulo_tf import Pendulo
import random
from _collections import deque
from inputs import Axis, KeysControl
from tools import draw_center_mass, draw_particles, draw_path_particles, lerp, lerp_v2, lerp_v3



class Cart():
    def __init__(self,
                 surface,
                 controller: Axis | KeysControl,
                 pos=(0., 0.),
                 color=(255, 255, 255),
                 center_mass_colors = ( (255,)*3, (0,)*3 ),
                 selected = False,
                 size = 30,
                 width = 4,
                 th0 = 0.,
                 linear_factor= 50, #50,
                 force_factor = 1.,
                 alive=True,
                 ):

        self.input = controller
        self.surface = surface
        self.pos = pos if isinstance(pos, list) else list(pos)
        self.color = color
        self.center_mass_colors = center_mass_colors
        self.selected = selected
        self.size = size if isinstance(size, tuple) else list(size) if isinstance(size, list) else (size, size)
        self.width = width

        self.th_target = math.pi
        self.th_tol = math.radians(30.)
        self.x_target = self.surface.get_width() / 2
        self.x_tol = 100.

        self.alive = alive
        self.ticks = 0
        self.ticks_since_death = 0
        self.cart_on_target = False
        self.pole_on_target = False
        self.steps_with_pole_on_target = 0
        self.steps_with_both_on_target = 0
        self.score = 0
        self.uncollected_score = 0
        self.reward = 0

        self.center_mass = 0.5
        self.last_input = 0.
        self.LINEAR_FACTOR = linear_factor
        self.FORCE_FACTOR = force_factor
        # self.model = Pendulo(1., 0.3, 5, x_damping=1, theta_damping=1, x0 = self.pos[0]/self.LINEAR_FACTOR, th0=th0, dt=1/60)
        self.model = Pendulo(1., .3, 5, 1., 1., x0 = self.pos[0]/self.LINEAR_FACTOR, th0=th0, dt=1/60)
        self.jet = pygame.image.load("assets/jet.png")
        # self.particles_colors = ((90, 90, 60), (90, 60, 90))
        self.trace_particles_colors = ((90, 90, 60), (90, 60, 90))
        self.highlight_particles_colors = ((200, 200, 60), self.color)

        self.N_trace = 120
        self.trace = deque(maxlen=self.N_trace)

    def saturate_pos(self) -> bool:
        original = (self.pos[0], self.pos[1])
        self.pos[0] = max(self.size[0] // 2, min(self.surface.get_width() - self.size[0] // 2, self.pos[0]))
        self.pos[1] = max(self.size[1] // 2, min(self.surface.get_height() - self.size[1] // 2, self.pos[1]))
        return self.pos[0] == original[0] and self.pos[1] == original[1]

    def set_pos(self, x=None, y=None) -> bool:
        if x is not None:
            self.pos[0] = x
        if y is not None:
            self.pos[1] = y
        return self.saturate_pos()

    def delta_pos(self, dx=0., dy=0.):
        self.pos[0] += dx
        self.pos[1] += dy
        self.saturate_pos()

    # def draw(self):
    #     pygame.draw.line(self.surface, self.color, (self.pos[0] - self.size[0] // 2, self.pos[1]), (self.pos[0] + self.size[1] // 2, self.pos[1]), self.width)
    #     pygame.draw.line(self.surface, self.color, (self.pos[0], self.pos[1] - self.size[1] // 2), (self.pos[0], self.pos[1] + self.size[1] // 2), self.width)
    #     if self.selected:
    #         pygame.draw.line(self.surface, self.color, (self.pos[0] - self.size[0] // 3, self.pos[1]), (self.pos[0], self.pos[1] + self.size[1] // 3),  width=self.width)
    #         pygame.draw.line(self.surface, self.color, (self.pos[0], self.pos[1] - self.size[1] // 3), (self.pos[0] + self.size[0] // 3, self.pos[1]),  width=self.width)
    #         pygame.draw.line(self.surface, self.color, (self.pos[0] + self.size[0] // 3, self.pos[1]), (self.pos[0], self.pos[1] + self.size[1] // 3),  width=self.width)
    #         pygame.draw.line(self.surface, self.color, (self.pos[0], self.pos[1] - self.size[1] // 3), (self.pos[0] - self.size[0] // 3, self.pos[1]),  width=self.width)

    def collect_score(self):
        x = self.uncollected_score
        self.uncollected_score = 0
        return x

    def step(self) -> bool:
        self.last_input = self.input.value
        self.reward = 0
        if self.alive:
            self.model.step(self.input.value*self.FORCE_FACTOR)
            if not self.set_pos(x=self.model.y[0][0]*self.LINEAR_FACTOR):
                self.alive = False

            self.cart_on_target = abs(self.pos[0] - self.x_target) < self.x_tol and self.alive
            self.pole_on_target = abs(math.fmod(self.model.theta - self.th_target, 2*math.pi)) < self.th_tol and self.alive

            self.steps_with_pole_on_target = self.steps_with_pole_on_target + 1 if self.pole_on_target else 0
            self.steps_with_both_on_target = self.steps_with_both_on_target + 1 if self.pole_on_target and self.cart_on_target else 0

            if self.steps_with_pole_on_target > 60:
                self.reward += 1 if self.steps_with_pole_on_target < 60*3 else 2

            if self.steps_with_both_on_target > 60:
                self.reward += 2 if self.steps_with_both_on_target < 60*3 else 6
            self.ticks += 1
        else:
            self.reward = -1
            self.ticks_since_death += 1

        self.score += self.reward
        self.uncollected_score += self.reward
        self.score = max(self.score, 0)
        return self.alive


    def feedback(self):
        if self.alive:
            if isinstance(self.input, Axis):
                l = self.model.linear_acceleration / 120
                r = -self.model.linear_acceleration / 120

                self.input.source.rumble(l*.05, r, 100)


    @property
    def pole_tip_pos(self):
        x = self.pos[0] + self.size[0] * math.sin(self.model.theta)
        y = self.pos[1] + self.size[0] * math.cos(self.model.theta)
        return x, y

    def draw(self):
        base_width = self.size[0]
        base_height = self.size[1]
        pole_length = base_width

        if self.alive:
            color = self.color
            pole_color = (int(self.color[0]*1.5), int(self.color[1]*1.5), int(self.color[2]*1.5))
            highlight_color = self.highlight_particles_colors[0]
            center_mass_colors = self.center_mass_colors
        else:
            c2 = (60, 60, 50)
            t = 0.85
            color = lerp_v3(self.color, c2, t)
            pole_color = color
            highlight_color = color
            center_mass_colors = (lerp_v3(self.center_mass_colors[0], c2, t), lerp_v3(self.center_mass_colors[1], c2, t))

        if self.cart_on_target and self.pole_on_target:
            pygame.draw.rect(self.surface, highlight_color, (self.pos[0] - base_width//2 - 2, self.pos[1] - base_height//2 - 2, base_width+4, base_height+4), self.width+2)
            draw_path_particles(self.surface, self.highlight_particles_colors[0], self.highlight_particles_colors[1], (self.pos[0] - base_width//2 - 2, self.pos[1]), (self.pos[0] + base_width//2 - 2, self.pos[1]), (base_height)//2, 20)

        pygame.draw.rect(self.surface, color, (self.pos[0] - base_width//2, self.pos[1] - base_height//2, base_width, base_height))


        intensity_pixels_gain = int(8 + 2*random.random()) * 20
        r_intensity = max(self.last_input*intensity_pixels_gain, 0.)
        l_intensity = max(-self.last_input*intensity_pixels_gain, 0.)
        jet_height = base_height * 2.75

        l_image = pygame.transform.smoothscale(pygame.transform.flip(self.jet, True, False), (l_intensity, jet_height))
        r_image = pygame.transform.smoothscale(self.jet, (r_intensity, jet_height))

        # pygame.draw.rect(self.surface, (255,0,0), (self.pos[0] + base_width//2, self.pos[1] - base_height//2, l_intensity, base_height), self.width)
        # pygame.draw.rect(self.surface, (255,0,0), (self.pos[0] - base_width//2-r_intensity, self.pos[1] - base_height//2, r_intensity, base_height), self.width)

        if self.alive:
            self.surface.blit(l_image, (self.pos[0] + base_width//2, self.pos[1] - jet_height//2, l_intensity, jet_height))
            self.surface.blit(r_image, (self.pos[0] - base_width//2-r_intensity, self.pos[1] - jet_height//2, r_intensity, jet_height))
            for i, pos in enumerate(self.trace):
                draw_particles(self.surface, self.trace_particles_colors[0], self.trace_particles_colors[1], pos, int(12 * i / self.N_trace), int(2 + 10 * i / 120))

        if self.pole_on_target:
            draw_pole(self.surface, highlight_color, self.pos, -self.model.theta+math.pi/2, pole_length+3, 14, center_mass=self.center_mass, center_mass_colors=center_mass_colors)
            th = -self.model.theta+math.pi/2
            start = self.pos
            end = (start[0]+pole_length*math.cos(th), start[1]+pole_length*math.sin(th))
            draw_path_particles(self.surface, (200, 200, 0), (30, 40, 30), start, end, 12, 30)
        draw_pole(self.surface, pole_color, self.pos, -self.model.theta+math.pi/2, pole_length, 8, center_mass=self.center_mass, center_mass_colors=center_mass_colors)

        draw_center_mass(self.surface, self.pos, colors=center_mass_colors)

        if self.alive:
            theta = -self.model.theta + math.pi/2
            self.trace.append((self.pos[0] + self.center_mass*pole_length * math.cos(theta), self.pos[1] + self.center_mass*pole_length * math.sin(theta)) )


def draw_pole(surface: pygame.surface,
              color: tuple[int, int, int],
              pos: tuple[int, int] | list[int, int],
              theta: float,
              length: float,
              width: int = 1,
              center_mass: float | None = None,
              center_mass_colors = ((255,)*3, (0,)*3),
              ):

    end_pos = (pos[0] + length * math.cos(theta), pos[1] + length * math.sin(theta))

    # pygame.draw.line(surface, color, pos, end_pos, width=width)
    draw_line_rect(surface, color, pos, end_pos, width=width)
    if center_mass is not None:
        mass_pos = (pos[0] + center_mass*length * math.cos(theta), pos[1] + center_mass*length * math.sin(theta))
        draw_center_mass(surface, mass_pos, colors=center_mass_colors)



def draw_line_rect(surface, color, pos, end_pos, width):
    y = float(end_pos[1] - pos[1])
    x = float(end_pos[0] - pos[0])
    h = math.sqrt(x**2 + y**2)

    cos_th = x / h
    sin_th = y / h
    w2 = width/2

    points = (
        (pos[0] + w2 * sin_th, pos[1] - w2 * cos_th),
        (pos[0] - w2 * sin_th, pos[1] + w2 * cos_th),
        (pos[0] - w2 * sin_th + x, pos[1] + w2 * cos_th + y),
        (pos[0] + w2 * sin_th + x, pos[1] - w2 * cos_th + y),

    )

    pygame.gfxdraw.filled_polygon(surface, points, color)
    pygame.gfxdraw.aapolygon(surface, points, color)
