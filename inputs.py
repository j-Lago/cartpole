import random
from typing import Callable
from functools import partial
import pygame



JOYBUTTON: dict[str, int] = {
    'x': 0,
    'c': 1,
    's': 2,
    't': 3,
    'select': 4,
    'PS': 5,
    'start': 6,
    'L3': 7,
    'R3': 8,
    'L1': 9,
    'R1': 10,
    'up': 11,
    'down': 12,
    'left': 13,
    'right': 14,
    'pad': 15,
}

class Axis():
    def __init__(self, source: pygame.joystick, channel, dead_zone: float = 0., initial_value = 0., normalization: Callable = lambda x: x):
        self.source = source
        self.channel = channel
        self.value = initial_value
        self.dead_zone = dead_zone
        self.normalization = normalization
        self.device_type = 'joystick'

    def update(self):
        self.value = self.normalization(remove_dead_zone(self.source.get_axis(self.channel), self.dead_zone))



class KeysControl():
    def __init__(self, source, key_left, key_right, key_intensity = None, initial_value = 0., normalization: Callable = lambda x: x):
        self.source = source
        self.key_left = key_left
        self.key_right = key_right
        self.key_intensity = key_intensity
        self.value = initial_value
        self.normalization = normalization
        self.device_type = 'keyboard'

    def update(self):
        keys = self.source.get_pressed()
        out = 0
        if keys[self.key_left]:
            out = -1
        if keys[self.key_right]:
            out = 1

        if self.key_intensity is not None:
            if keys[self.key_intensity]:
                out *= 0.5

        self.value = self.normalization(out)


class IAControl:
    def __init__(self, initial_value = 0., normalization: Callable = lambda x: x):
        self.value = initial_value
        self.normalization = normalization
        self.device_type = 'IA'


    def update(self):
        self.value += self.normalization((random.random()*2-1)*0.6)
        self.value = max(-1., min(self.value, 1.))



def remove_dead_zone(x, dead_zone):
    if abs(x) < dead_zone:
        return 0.
    return x
