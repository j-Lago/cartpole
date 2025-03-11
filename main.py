import pygame
import sys
from player import Player, Cart
from inputs import Axis
import math
import random

import os
os.environ['SDL_JOYSTICK_HIDAPI_PS4_RUMBLE'] = '1'


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60
INPUT_STEP = 10

paused = False
name = 'Controllab'

pygame.init()
pygame.display.set_caption(name)
pygame.mouse.set_visible(False)

pygame.mixer.init()
sounds = {
    'laser': pygame.mixer.Sound('assets/laser.wav'),
    'pistol': pygame.mixer.Sound('assets/pistol.wav'),
    'reload': pygame.mixer.Sound('assets/reload.wav'),
    'mario': pygame.mixer.Sound('assets/mario.wav'),
    'coin': pygame.mixer.Sound('assets/coin.wav'),
    'jet': pygame.mixer.Sound('assets/jet.wav'),
    'jet2': pygame.mixer.Sound('assets/jet2.wav'),
    'beep': pygame.mixer.Sound('assets/beep.wav'),
}

joystick = pygame.joystick.Joystick(0)
joystick.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

font_norm = pygame.font.SysFont('Consolas', 22)
font_med = pygame.font.SysFont('Consolas', 120)
font_big = pygame.font.SysFont('Consolas', 240)

players = dict()



def pause():
    global paused, name
    paused = not paused
    pygame.display.set_caption(name if not paused else f'{name} (PAUSED)')


def reset():
    global players, paused, name, sounds
    y_sup = 0.35
    y_inf = 0.75
    color_sup = (60, 120, 100)
    color_inf = (120, 60, 100)
    players = {
        'targuet_p6': Cart(screen, (SCREEN_WIDTH // 2, int(SCREEN_HEIGHT*y_sup)), color=color_sup, width=3, size=(180, 21), th0=math.pi, alive=False),
        'targuet_p7': Cart(screen, (SCREEN_WIDTH // 2, int(SCREEN_HEIGHT*y_inf)), color=color_inf, width=3, size=(180, 21), th0=math.pi, alive=False),
        'p6': Cart(screen, (SCREEN_WIDTH *2 // 3, int(SCREEN_HEIGHT*y_sup)), color=color_sup, width=3, size=(180, 21), th0=math.pi+random.random()*.1),
        'p7': Cart(screen, (SCREEN_WIDTH *0.3 // 3, int(SCREEN_HEIGHT*y_inf)), color=color_inf, width=3, size=(180, 21), th0=0.),
        'mouse': Player(screen, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), color=(90, 80, 70), size=100, width=1),
    }
    paused = False
    pygame.display.set_caption(name if not paused else f'{name} (PAUSED)')
    sounds['beep'].play()
    joystick.rumble(1.0, 1.0, 500)

reset()
# Inicializando o relógio do Pygame para controlar o frame rate
clock = pygame.time.Clock()



L2 = Axis(source=joystick, channel=4, normalization=lambda x: (x+1.0)*0.5)
R2 = Axis(source=joystick, channel=5, normalization=lambda x: (x+1.0)*0.5)
lx = Axis(source=joystick, channel=0, dead_zone=0.05)
ly = Axis(source=joystick, channel=1, dead_zone=0.05)
rx = Axis(source=joystick, channel=2, dead_zone=0.05)
ry = Axis(source=joystick, channel=3, dead_zone=0.05)

jet_r = sounds['jet'].play(loops=-1)
jet_l = sounds['jet2'].play(loops=-1)

# Loop principal do jogo
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.JOYAXISMOTION:
            if not paused:
                lx.update()
                ly.update()
                rx.update()
                ry.update()
                L2.update()
                R2.update()

        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                reset()
            if keys[pygame.K_SPACE]:
                pause()


        if event.type == pygame.JOYBUTTONDOWN:
            #  0: x
            #  1: c
            #  2: s
            #  3: t
            #  4: select
            #  5: ps
            #  6: start
            #  7: L3
            #  8: R3
            #  9: L1
            # 10: R1
            # 11: up
            # 12: down
            # 13: left
            # 14: right
            # 15: pad
            if joystick.get_button(6):
                pause()


            if joystick.get_button(5):
                reset()


            if not paused:
                if joystick.get_button(0):
                    sounds['pistol'].play()
                    joystick.rumble(1.0, 1.0, 100)
                if joystick.get_button(1):
                    sounds['laser'].play()
                if joystick.get_button(2):
                    sounds['reload'].play()
                if joystick.get_button(3):
                    sounds['mario'].play()
                if joystick.get_button(12):
                    sounds['coin'].play()




    def remove_dead_zone(value):
        DEAD_ZONE = 0.05
        if abs(value) < DEAD_ZONE:
            value = 0.
        return value

    SENSITIVITY = 2


    # players['p1'].delta_pos(dx=INPUT_STEP * lx.value, dy=INPUT_STEP * ly.value)
    # players['p3'].delta_pos(dx=INPUT_STEP * (R2.value-L2.value), dy=0.)


    # players['p6'].delta_pos(dx=INPUT_STEP * (R2.value-L2.value)*SENSITIVITY)

    if not paused:
        # if joystick.get_button(13):
        #     rx.value = -1.
        # elif joystick.get_button(14):
        #     rx.value = 1.
        # else:
        #     rx.update()

        if players['p6'].alive:
            players['p6'].step(INPUT_STEP * rx.value*SENSITIVITY)
            if not players['p6'].alive:
                sounds['mario'].play()

        if players['p7'].alive:
            players['p7'].step(INPUT_STEP * lx.value*SENSITIVITY)
            if not players['p7'].alive:
                sounds['mario'].play()

        r_volume = abs(rx.value) if players['p6'].alive else 0.
        l_volume = abs(lx.value) if players['p7'].alive else 0.

    jet_r.set_volume(r_volume if not paused else 0.)
    jet_l.set_volume(l_volume if not paused else 0.)

    l = players['p6'].model.linear_acceleration / 120
    r = -players['p6'].model.linear_acceleration / 120
    if not paused and players['p6'].alive:
        joystick.rumble(l*.05, r, 100)
    # print(players['p6'].model.linear_acceleration)





    mouse_x, mouse_y = pygame.mouse.get_pos()
    players['mouse'].set_pos(mouse_x, mouse_y)

    # delta12_x = (players['p1'].pos[0] - players['p2'].pos[0])
    # delta12_y = (players['p1'].pos[1] - players['p2'].pos[1])
    # players['p2'].delta_pos(dx=delta12_x * INPUT_STEP * .01, dy=delta12_y * INPUT_STEP * .01)
    #
    # delta34_x = (players['p3'].pos[0] - players['p4'].pos[0])
    # delta4_y = (players['p3'].pos[1]-players['p4'].pos[1])
    # players['p4'].delta_pos(dx=delta34_x * INPUT_STEP * .01, dy=delta4_y * INPUT_STEP * .01)


    # render
    if not paused:
        screen.fill((25, 25, 25))
        for player in players.values():
            player.draw()
        fps = clock.get_fps()
    else:
        paused_text = font_big.render('PAUSED', True, (150, 150, 150))
        screen.blit(paused_text, ((SCREEN_WIDTH - paused_text.get_width()) // 2, (SCREEN_HEIGHT - paused_text.get_height()) // 2))



    def scolor(value: float, tol=0.01) -> tuple[int, int, int]:
        if value + tol < 0:
            return 200, 125, 125
        if value - tol > 0:
            return 100, 200, 150
        return 200, 200, 150

    hud_color = (100, 90, 70)

    # players['p7'].score = 1234567890
    fps_text = font_norm.render(f"{fps:.1f}", True, hud_color)
    sup_score = font_med.render(f"{players['p6'].score:>10d}", True, players['p6'].color)
    inf_score = font_med.render(f"{players['p7'].score:>10d}", True, players['p7'].color)
    mouse_text = font_norm.render(f'({mouse_x:4d},{mouse_y:4d})', True, hud_color)


    if not paused:
        screen.blit(fps_text, (10, SCREEN_HEIGHT-30))
        screen.blit(mouse_text, (30, 30))
        screen.blit(sup_score, (SCREEN_WIDTH-670, 10))
        screen.blit(inf_score, (SCREEN_WIDTH-670,SCREEN_HEIGHT-140))

    pygame.display.flip()
    clock.tick(FPS)
    