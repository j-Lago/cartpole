from typing import Dict, Any

import pygame
import sys
from player import Cart
from inputs import Axis, KeysControl, IAControl, JOYBUTTON
import math
import random
from enum import Enum
import assets
from assets import colors as cols
from tools import lerp_v3
import json


import os
os.environ['SDL_JOYSTICK_HIDAPI_PS4_RUMBLE'] = '1'


class GAMESTATE(Enum):
    PRE_INIT = -1
    RUN = 1
    PAUSED = 0
    TIMEOUT = -2
    GAME_OVER = -3


class Game():
    def __init__(self, name: str, window_size: tuple[int, int] | None, fps: int, sounds: dict, fonts: dict, images: dict, game_duration: float = 30., max_power: float = 18., save_file: str = 'meta/save.json'):

        self.fps = fps
        self.duration = game_duration
        self.MAX_POWER = max_power
        self.save_file_name =  save_file

        pygame.init()
        pygame.mouse.set_visible(False)


        if window_size is None:
            screen_info = pygame.display.Info()
            self.screen = pygame.display.set_mode((screen_info.current_w, screen_info.current_h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(window_size)


        pygame.display.set_caption(name)
        self.mixer = pygame.mixer
        self.mixer.init()

        self.joysticks = dict()
        for i in range(pygame.joystick.get_count()):
            key = f'p{i+1}'
            self.joysticks[key] = pygame.joystick.Joystick(i)
            self.joysticks[key].init()

        self.axes = dict()
        key = 'p1'
        if key in self.joysticks.keys():
            self.axes[key] = Axis(source=self.joysticks[key], channel=2, dead_zone=0.05)
        else:
            self.axes[key] = KeysControl(source=pygame.key, key_left=pygame.K_LEFT, key_right=pygame.K_RIGHT, key_intensity=pygame.K_RCTRL)

        key = 'p2'
        if key in self.joysticks.keys():
            self.axes[key] = Axis(source=self.joysticks[key], channel=2, dead_zone=0.05)
        else:
            # self.axes[key] = KeysControl(source=pygame.key, key_left=pygame.K_a, key_right=pygame.K_d, key_intensity=pygame.K_SPACE)
            self.axes[key] = IAControl()


        # assets
        self.sounds = load_sounds(sounds)
        self.fonts = load_fonts(fonts)
        self.images = load_images(images)

        # -- reset --------------------------------------
        self.clock = None
        self.time = None
        self.paused_time = None
        self.state = None
        self.players = dict()
        self.inputs = dict()
        # -- reset --------------------------------------
        self.reset()

        try:
            with open(self.save_file_name, 'r') as arquivo_json:
                data = json.load(arquivo_json)
            if 'best_score' in data.keys():
                self.best_score = data['best_score']
            else:
                self.best_score = 0
        except:
            self.best_score = 0


        self.loop()

    def reset(self):
        self.clock = pygame.time.Clock()
        self.time = 0.
        self.paused_time = 0.
        self.state = GAMESTATE.PRE_INIT
        y_sup = 0.35
        y_inf = 0.78
        self.players = {
            'target_p1': Cart(self.screen, None, (self.screen_center[0]     , int(self.screen_height*y_sup)), color=cols['p1'], width=3, size=assets.sizes['cart'], th0=math.pi, alive=False),
            'target_p2': Cart(self.screen, None, (self.screen_center[0]     , int(self.screen_height*y_inf)), color=cols['p2'], width=3, size=assets.sizes['cart'], th0=math.pi, alive=False),
            'p1'       : Cart(self.screen, self.axes['p1'], (self.screen_width * 2 // 3, int(self.screen_height*y_sup)), color=cols['p1'], width=3, size=assets.sizes['cart'], th0=math.pi+random.random()*.1, force_factor=self.MAX_POWER),
            'p2'       : Cart(self.screen, self.axes['p2'], (self.screen_width * 1 // 3, int(self.screen_height*y_inf)), color=cols['p2'], width=3, size=assets.sizes['cart'], th0=math.pi+random.random()*.1, force_factor=self.MAX_POWER),
        }

        for axis in self.axes.values():
            axis.value  = 0.

        pygame.event.clear()


    def loop(self):
        while True:

            if self.state == GAMESTATE.RUN:
                self.time += 1/self.fps  # não usar o tempo real do sistema permite acelerar simulação para treinamento
                if self.time > self.duration:
                    self.timeout()
                if not self.players['p1'].alive and not self.players['p2'].alive:
                    self.game_over()
            else:
                self.paused_time += 1/self.fps
                if self.state == GAMESTATE.PRE_INIT:
                    if self.paused_time > 0.5:
                        self.start()
                    else:
                        self.pre_init()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_to_file()
                    pygame.quit()
                    sys.exit()

                if event.type in [pygame.JOYBUTTONDOWN, pygame.KEYDOWN]:
                    for joystick in self.joysticks.values():
                        if joystick.get_button(JOYBUTTON['start']): self.pause()
                        if joystick.get_button(JOYBUTTON['PS']): self.reset()

                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_RETURN]: self.pause()
                    if keys[pygame.K_ESCAPE]: self.reset()



            if self.state == GAMESTATE.RUN:
                self.process_inputs()
                self.simulate()
                self.process_feedback()
            if self.state == GAMESTATE.RUN or self.state == GAMESTATE.PRE_INIT:
                self.draw()
            self.process_sounds()

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
        pass

    def save_to_file(self):
        save = {'best_score': self.best_score}
        with open(self.save_file_name, 'w') as arquivo_json:
            json.dump(save, arquivo_json)

    def game_over(self):
        self.state = GAMESTATE.GAME_OVER
        text = self.fonts['big'].render(f"GAME OVER", True, cols['hud'])
        self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))
        self.save_score()

    def timeout(self):
        self.state = GAMESTATE.TIMEOUT
        self.sounds['whistle'].play()
        text = self.fonts['big'].render(f"TIMEOUT", True, cols['hud'])
        self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))
        self.save_score()

    def start(self):
        self.state = GAMESTATE.RUN
        self.sounds['beep'].play()
        self.sounds['jet_r'].play(loops=-1)
        self.sounds['jet_l'].play(loops=-1)

    def pause(self):
        if self.state == GAMESTATE.RUN:
            self.state = GAMESTATE.PAUSED
            text = self.fonts['big'].render(f"PAUSED", True, cols['hud'])
            self.screen.blit(text, (self.screen_center[0] - text.get_width() // 2, self.screen_center[1] - text.get_height() // 2))
        elif self.state == GAMESTATE.PAUSED:
            self.state = GAMESTATE.RUN
        else:
            self.reset()

    def save_score(self):
        for player in self.players.values():
            if player.score > self.best_score:
                self.best_score = player.score

    def process_inputs(self):
        for axis in self.axes.values():
            axis.update()

        self.inputs['p1'] = self.axes['p1'].value * self.MAX_POWER
        self.inputs['p2'] = self.axes['p2'].value * self.MAX_POWER

    def simulate(self):

        if self.players['p1'].alive:
            self.players['p1'].step()
            if not self.players['p1'].alive:
                self.sounds['death'].play()

        if self.players['p2'].alive:
            self.players['p2'].step()
            if not self.players['p2'].alive:
                self.sounds['death'].play()


    def process_sounds(self):
        r_volume = abs(self.players['p1'].input.value) if self.players['p1'].alive else 0.
        l_volume = abs(self.players['p2'].input.value) if self.players['p2'].alive else 0.

        self.sounds['jet_r'].set_volume(r_volume if self.state == GAMESTATE.RUN else 0.)
        self.sounds['jet_l'].set_volume(l_volume if self.state == GAMESTATE.RUN else 0.)

    def process_feedback(self):
        if self.state == GAMESTATE.RUN:
            for player in self.players.values():
                player.feedback()

    def draw(self):
        self.screen.fill(cols['bg'])
        for player in self.players.values():
            player.draw()


        dcols = { key: cols[key] if self.players[key].alive else lerp_v3(cols[key], (60, 60, 50), 0.85) for key in ['p1', 'p2']}


        fps = self.clock.get_fps()
        text_fps = self.fonts['small'].render(f"{fps:.1f}", True, cols['hud'])
        text_timer = self.fonts['normal'].render(f"{self.duration-self.time:.1f}", True, cols['timer'])
        text_timer_label = self.fonts['small'].render(f"TIMER", True, cols['timer'])
        text_p1 = self.fonts['medium'].render(f"{self.players['p1'].score:>10d}", True, dcols['p1'])
        text_p2 = self.fonts['medium'].render(f"{self.players['p2'].score:>10d}", True, dcols['p2'])
        text_best = self.fonts['normal'].render(f"{self.best_score:<10d}", True, cols['best_score'])
        text_p1_label = self.fonts['small'].render(f"P1 SCORE", True, dcols['p1'])
        text_p2_label = self.fonts['small'].render(f"P2 SCORE", True, dcols['p2'])
        text_best_label = self.fonts['small'].render(f"BEST SCORE", True, cols['best_score'])
        self.screen.blit(text_fps, (30, 30))
        self.screen.blit(text_p1, (self.screen_width - text_p1.get_width() - 30, 40))
        self.screen.blit(text_p2, (self.screen_width - text_p2.get_width() - 30, self.screen_height - 140))
        self.screen.blit(text_best, ( 30, self.screen_center[1] - text_center(text_best)[1]))
        self.screen.blit(text_timer, (self.screen_width - text_timer.get_width() - 30, self.screen_center[1] - text_center(text_timer)[1]))
        self.screen.blit(text_timer_label, (self.screen_width - text_timer_label.get_width()-30, self.screen_center[1] - text_center(text_timer_label)[1]-60))
        self.screen.blit(text_p1_label, (self.screen_width - text_p1_label.get_width() - 30, 150))
        self.screen.blit(text_p2_label, (self.screen_width - text_p2_label.get_width() - 30, self.screen_height - 170))
        self.screen.blit(text_best_label, (30, self.screen_center[1] - text_center(text_best_label)[1]-60))

        if self.state == GAMESTATE.PRE_INIT:
            text = self.fonts['big'].render(f"START", True, cols['hud'])
            self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))


def load_sounds(description: dict) -> dict:
    return {k: pygame.mixer.Sound(v) for k, v in description.items()}


def load_fonts(description: dict) -> dict:
    return {k: pygame.font.SysFont(*v) for k, v in description.items()}


def load_images(description: dict) -> dict:
    return {k: pygame.image.load(v) for k, v in description.items()}


def text_center(text) -> tuple[int, int]:
    return text.get_width() // 2, text.get_height() // 2


