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
from copy import copy, deepcopy


import os
os.environ['SDL_JOYSTICK_HIDAPI_PS4_RUMBLE'] = '1'


class GAMESTATE(Enum):
    PRE_INIT = -1
    RUN = 1
    PAUSED = 0
    TIMEOUT = -2
    GAME_OVER = -3


class Game():
    def __init__(self, name: str, window_size: tuple[int, int] | None, fps: int, sounds: dict, fonts: dict, images: dict, game_duration: float = 30., max_power: float = 18., save_file: str = 'meta/save.json', DO_NOT_RENDER: bool = False, STEP_BY_STEP: bool = False):

        self.fps = fps
        self.duration = game_duration
        self.MAX_POWER = max_power
        self.save_file_name =  save_file
        self.DO_NOT_RENDER = DO_NOT_RENDER
        self.STEP_BY_STEP = STEP_BY_STEP
        self.screen_shake_disable = True

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
            self.axes[key] = KeysControl(source=pygame.key, key_left=pygame.K_LEFT, key_right=pygame.K_RIGHT, key_intensity=pygame.K_RALT)

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
            if 'best_score' in data.keys() and 'best_score_device' in data.keys():
                self.best_score = data['best_score']
                self.best_score_device = data['best_score_device']
            else:
                self.best_score = 0
                self.best_score_device = None

        except:
            self.best_score = 0
            self.best_score_device = None

        self.pre_init()
        if not self.STEP_BY_STEP:
            self.loop()

    def reset(self):
        self.clock = pygame.time.Clock()
        self.time = 0.
        self.paused_time = 0.
        self.state = GAMESTATE.PRE_INIT
        y_sup = 0.35
        y_inf = 0.78
        self.npcs = {
                'target_p1': Cart(self.screen, None, (self.screen_center[0]     , int(self.screen_height*y_sup)), color=cols['p1'], width=3, size=assets.sizes['cart'], th0=math.pi, alive=False),
                'target_p2': Cart(self.screen, None, (self.screen_center[0]     , int(self.screen_height*y_inf)), color=cols['p2'], width=3, size=assets.sizes['cart'], th0=math.pi, alive=False),
            }
        if not self.DO_NOT_RENDER and not self.STEP_BY_STEP:
            self.players = {
                'p1'       : Cart(self.screen, self.axes['p1'], (self.screen_width * 2 // 3, int(self.screen_height*y_sup)), color=cols['p1'], width=3, size=assets.sizes['cart'], th0=math.pi*0+random.random()*.1, force_factor=self.MAX_POWER),
                'p2'       : Cart(self.screen, self.axes['p2'], (self.screen_width * 1 // 3, int(self.screen_height*y_inf)), color=cols['p2'], width=3, size=assets.sizes['cart'], th0=math.pi*0+random.random()*.1, force_factor=self.MAX_POWER),
            }
        else:
            self.players = {
                'p2'       : Cart(self.screen, self.axes['p2'], (self.screen_width * 1 // 3, int(self.screen_height*y_inf)), color=cols['p2'], width=3, size=assets.sizes['cart'], th0=math.pi*0+random.random()*.1, force_factor=self.MAX_POWER),
            }

        for axis in self.axes.values():
            axis.value  = 0.

        pygame.event.clear()


    def inc_time(self):
        if self.state == GAMESTATE.RUN:
            self.time += 1/self.fps  # não usar o tempo real do sistema permite acelerar simulação para treinamento
            if self.time > self.duration:
                self.timeout()

    def step(self, axis_input):
        if self.state == GAMESTATE.PRE_INIT:
            self.pre_init()
        self.inc_time()
        for key, player in self.players.items():
            player.input.value = axis_input * self.MAX_POWER
            print(player.input.value)
        self.simulate()

        if self.all_dead():
            self.game_over()

        if (self.state == GAMESTATE.RUN or self.state == GAMESTATE.PRE_INIT) and not self.DO_NOT_RENDER:
            self.draw()

        if not self.DO_NOT_RENDER:
            pygame.display.flip()
            self.clock.tick(self.fps)

    def loop(self):
        while True:

            if self.state == GAMESTATE.RUN:
                self.inc_time()

                all_dead = True
                for player in self.players.values():
                    all_dead = all_dead and not player.alive
                if all_dead:
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
                    if keys[pygame.K_k]: self.screen_shake_disable = not self.screen_shake_disable


            if self.state == GAMESTATE.RUN:
                self.process_inputs()
                self.simulate()
                self.process_feedback()
            if (self.state == GAMESTATE.RUN or self.state == GAMESTATE.PRE_INIT) and not self.DO_NOT_RENDER:
                self.draw()
            if not self.DO_NOT_RENDER:
                self.process_sounds()

            if not self.DO_NOT_RENDER:
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
        if self.STEP_BY_STEP:
            self.state = GAMESTATE.RUN

    def save_to_file(self):
        save = {'best_score': self.best_score, 'best_score_device': self.best_score_device}
        with open(self.save_file_name, 'w') as arquivo_json:
            json.dump(save, arquivo_json)

    def game_over(self):
        self.state = GAMESTATE.GAME_OVER
        if not self.DO_NOT_RENDER:
            text = self.fonts['big'].render(f"GAME OVER", True, cols['hud'])
            self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))
        self.save_score()

    def timeout(self):
        self.state = GAMESTATE.TIMEOUT
        if not self.DO_NOT_RENDER:
            self.sounds['whistle'].play()
            text = self.fonts['big'].render(f"TIMEOUT", True, cols['hud'])
            self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))
        self.save_score()

    def start(self):
        self.state = GAMESTATE.RUN
        if not self.DO_NOT_RENDER:
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
        print(f'{self.state} (', end='')
        for key, player in self.players.items():
            if not key.startswith('target_'):
                print(f'{key}: {player.score}', end=', ')
                if player.score > self.best_score:
                    self.best_score = player.score
                    self.best_score_device = player.input.device_type
        print('\b\b)')
        if self.DO_NOT_RENDER:
            self.reset()

    def process_inputs(self):
        for axis in self.axes.values():
            axis.update()
        for key, player in self.players.items():
            self.inputs[key] = self.axes[key].value * self.MAX_POWER

    def simulate(self):
        for key, player in self.players.items():
            player.step()
            if not player.alive and player.ticks_since_death == 0 and not self.DO_NOT_RENDER:
                self.sounds['death'].play()

    def all_dead(self) -> bool:
        all_dead = True
        for player in self.players.values():
            all_dead = all_dead and not player.alive
        return all_dead


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
        for npc in self.npcs.values():
            npc.draw()

        for player in self.players.values():
            player.draw()


        dcols = { key: cols[key] if self.players[key].alive else lerp_v3(cols[key], (60, 60, 50), 0.85) for key in self.players.keys()}


        fps = self.clock.get_fps()
        text_fps = self.fonts['small'].render(f"{fps:.1f}", True, cols['hud'])
        text_timer = self.fonts['normal'].render(f"{self.duration-self.time:.1f}", True, cols['timer'])
        text_timer_label = self.fonts['small'].render(f"TIMER", True, cols['timer'])
        if 'p1' in self.players.keys():
            text_p1 = self.fonts['medium'].render(f"{self.players['p1'].score:>10d}", True, dcols['p1'])
        else:
            dcols['p1'] = (60, 60, 50)
            text_p1 = self.fonts['medium'].render(f"{0:>10d}", True, dcols['p1'])
        text_p2 = self.fonts['medium'].render(f"{self.players['p2'].score:>10d}", True, dcols['p2'])
        text_best = self.fonts['normal'].render(f"{self.best_score:<10d}", True, cols['best_score'])
        text_p1_label = self.fonts['small'].render(f"P1 SCORE", True, dcols['p1'])
        text_p2_label = self.fonts['small'].render(f"P2 SCORE", True, dcols['p2'])
        text_best_label = self.fonts['small'].render(f"BEST SCORE", True, cols['best_score'])
        text_best_device = self.fonts['tiny'].render(f"{self.best_score_device}", True, cols['best_score'])
        self.screen.blit(text_fps, (30, 30))
        self.screen.blit(text_p1, (self.screen_width - text_p1.get_width() - 30, 40))
        self.screen.blit(text_p2, (self.screen_width - text_p2.get_width() - 30, self.screen_height - 140))
        self.screen.blit(text_best, ( 30, self.screen_center[1] - text_center(text_best)[1]))
        self.screen.blit(text_timer, (self.screen_width - text_timer.get_width() - 30, self.screen_center[1] - text_center(text_timer)[1]))
        self.screen.blit(text_timer_label, (self.screen_width - text_timer_label.get_width()-30, self.screen_center[1] - text_center(text_timer_label)[1]-50))
        self.screen.blit(text_p1_label, (self.screen_width - text_p1_label.get_width() - 30, 150))
        self.screen.blit(text_p2_label, (self.screen_width - text_p2_label.get_width() - 30, self.screen_height - 170))
        self.screen.blit(text_best_label, (30, self.screen_center[1] - text_center(text_best_label)[1]-50))
        self.screen.blit(text_best_device, (30, self.screen_center[1] - text_center(text_best_device)[1]+40))


        if self.state == GAMESTATE.PRE_INIT:
            text = self.fonts['big'].render(f"START", True, cols['hud'])
            self.screen.blit(text, (self.screen_center[0] - text_center(text)[0], self.screen_center[1] - text_center(text)[1]))

        if self.screen_shake_disable and self.state == GAMESTATE.RUN and not self.all_dead():
                shake = 0.
                for player in self.players.values():
                    if player.alive:
                        shake += abs(player.input.value)
                    elif player.ticks_since_death < 0.2 * self.fps:
                        shake += 10
                shake = shake / len(self.players) * 5
                screen = copy(self.screen)
                self.screen.fill(cols['bg'])
                self.screen.blit(screen, (random.random()*shake, random.random()*shake))

def load_sounds(description: dict) -> dict:
    return {k: pygame.mixer.Sound(v) for k, v in description.items()}


def load_fonts(description: dict) -> dict:
    return {k: pygame.font.SysFont(*v) for k, v in description.items()}


def load_images(description: dict) -> dict:
    return {k: pygame.image.load(v) for k, v in description.items()}


def text_center(text) -> tuple[int, int]:
    return text.get_width() // 2, text.get_height() // 2


