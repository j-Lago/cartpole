from typing import Dict, Any

import pygame
import sys
from player import Player, Cart
from inputs import Axis
import math
import random
from enum import Enum
import assets
from assets import colors as cols



import os
os.environ['SDL_JOYSTICK_HIDAPI_PS4_RUMBLE'] = '1'



class GAMESTATE(Enum):
    PRE_INIT = -1
    RUN = 1
    PAUSED = 0
    TIMEOUT = -2


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


class Game():
    def __init__(self, name: str, width: int, height: int, fps: int, sounds: dict, fonts: dict, images: dict, game_duration: float = 30.):

        self.fps = fps
        self.duration = game_duration

        pygame.init()
        pygame.mouse.set_visible(False)

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(name)
        self.mixer = pygame.mixer
        self.mixer.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.axes = {
            'L2': Axis(source=self.joystick, channel=4, normalization=lambda x: (x+1.0)*0.5),
            'R2': Axis(source=self.joystick, channel=5, normalization=lambda x: (x+1.0)*0.5),
            'lx': Axis(source=self.joystick, channel=0, dead_zone=0.05),
            'ly': Axis(source=self.joystick, channel=1, dead_zone=0.05),
            'rx': Axis(source=self.joystick, channel=2, dead_zone=0.05),
            'ry': Axis(source=self.joystick, channel=3, dead_zone=0.05),
        }

        # assets
        self.sounds = load_sounds(sounds)
        self.fonts = load_fonts(fonts)
        self.images = load_images(images)


        # -- reset --------------------------------------
        self.clock = pygame.time.Clock()
        self.time = 0.
        self.paused_time = 0.
        self.state = GAMESTATE.PRE_INIT
        # -- reset --------------------------------------

        self.loop()

    def reset(self):
        self.clock = pygame.time.Clock()
        self.time = 0.
        self.paused_time = 0.
        self.state = GAMESTATE.PRE_INIT


    def loop(self):
        while True:

            if self.state == GAMESTATE.RUN:
                self.time += 1/self.fps  # não usar o tempo real do sistema permite acelerar simulação para treinamento
                if self.time > self.duration:
                    self.timeout()
            else:
                self.paused_time += 1/self.fps
                if self.state == GAMESTATE.PRE_INIT:
                    if self.paused_time > 0.5:
                        self.start()
                    else:
                        self.pre_init()



            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.JOYAXISMOTION:
                    if self.state == GAMESTATE.RUN:
                        for axis in self.axes.values():
                            axis.update()

                if event.type == pygame.JOYBUTTONDOWN:
                    if self.joystick.get_button(JOYBUTTON['start']): self.pause()
                    if self.joystick.get_button(JOYBUTTON['PS']): self.reset()

            if self.state == GAMESTATE.RUN:
                self.draw()

            pygame.display.flip()
            self.clock.tick(self.fps)

    @property
    def screen_center(self) -> tuple[int, int]:
        return self.screen.get_width() // 2, self.screen.get_height() // 2

    @property
    def screen_width(self) -> int:
        return self.screen.get_width()

    @property
    def screen_height(self) -> int:
        return self.screen.get_height()

    def pre_init(self):
        self.screen.fill(cols['bg'])
        text = self.fonts['medium'].render(f"START", True, cols['hud'])
        self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))

    def timeout(self):
        self.state = GAMESTATE.TIMEOUT
        self.sounds['whistle'].play()
        text = self.fonts['medium'].render(f"TIMEOUT", True, cols['hud'])
        self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))

    def start(self):
        self.state = GAMESTATE.RUN
        self.sounds['beep'].play()

    def pause(self):
        if self.state == GAMESTATE.RUN:
            self.state = GAMESTATE.PAUSED
            text = self.fonts['medium'].render(f"START", True, cols['hud'])
            self.screen.blit(text, (self.screen_center[0] - text.get_width() // 2, self.screen_center[1] - text.get_height() // 2))
        elif self.state == GAMESTATE.PAUSED:
            self.state = GAMESTATE.RUN
        else:
            self.reset()

    def draw(self):
        self.screen.fill(cols['bg'])
        text = self.fonts['small'].render(f"({self.axes['rx'].value:+.2f}, {self.axes['ry'].value:+.2f})", True, cols['hud'])
        timer = self.fonts['normal'].render(f"{self.duration-self.time:.1f}", True, cols['timer'])

        self.screen.blit(text, (30, 30))
        self.screen.blit(timer, (self.screen_width - timer.get_width() - 30, self.screen_center[1] - text_center(timer)[1]))


def load_sounds(description: dict) -> dict:
    return {k: pygame.mixer.Sound(v) for k, v in description.items()}


def load_fonts(description: dict) -> dict:
    return {k: pygame.font.SysFont(*v) for k, v in description.items()}


def load_images(description: dict) -> dict:
    return {k: pygame.image.load(v) for k, v in description.items()}


def text_center(text) -> tuple[int, int]:
    return text.get_width() // 2, text.get_height() // 2


if __name__ == '__main__':
    game = Game('CartPole', 1600, 900, 60, assets.sounds, assets.fonts, assets.images, 10)