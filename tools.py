import random
import pygame
import math

def draw_center_mass(surface, pos, radius=15, colors=((255, 255, 255), (0, 0, 0))):
    pygame.draw.circle(surface, colors[0], pos, radius, draw_bottom_right=True, draw_top_left=True)
    pygame.draw.circle(surface, colors[1], pos, radius, draw_bottom_left=True, draw_top_right=True)


def draw_particles(surface, color1, color2, pos, max_radius, density):
    for _ in range(density):
        angle = random.uniform(0, 2 * 3.14159)
        radius = random.gauss(0, max_radius)
        x = int(pos[0] + radius * math.cos(angle))
        y = int(pos[1] + radius * math.sin(angle))
        if color1 is None or color2 is None:
            pcolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            pcolor = lerp_v3(color1, color2, random.random())
        pygame.draw.circle(surface, pcolor, (x,y), 1)


def draw_path_particles(surface, color1, color2, start_pos, end_pos, max_radius, density):
    for _ in range(density):
        t = random.uniform(0., 1.)
        center = lerp_v2(start_pos, end_pos, t)
        draw_particles(surface, color1, color2, center, max_radius, density)


def lerp(a, b, t):
    return a + (b-a)*t


def lerp_v2(p0, p1, t):
    return lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t)


def lerp_v3(p0, p1, t):
    return lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t), lerp(p0[2], p1[2], t)