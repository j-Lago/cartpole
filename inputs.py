from typing import Callable
from functools import partial
import pygame


class Axis():
    def __init__(self, source: pygame.joystick, channel, dead_zone: float = 0., initial_value = 0., normalization: Callable = lambda x: x):
        self.source = source
        self.channel = channel
        self.value = initial_value
        self.dead_zone = dead_zone
        self.normalization = normalization

    def update(self):
        self.value = self.normalization(remove_dead_zone(self.source.get_axis(self.channel), self.dead_zone))


def remove_dead_zone(x, dead_zone):
    if abs(x) < dead_zone:
        return 0.
    return x
