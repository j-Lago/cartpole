import math
import random
from random import random as rand, randint
import pygame

class Particle():
    def __init__(self, pos: tuple[float, float], vel: tuple[float, float], dt, decay=.999999, lifetime=-1, alive=True, linear_factor=1):
        self.x = pos[0]
        self.y = -pos[1]
        self.vel_x = vel[0]
        self.vel_y = vel[1]
        self.ticks = 0
        self.g = -9.81 * linear_factor
        self.decay = decay
        self.dt = dt
        self.lifetime = lifetime
        self.alive = alive

    @property
    def pos(self):
        return self.x, -self.y

    @property
    def vel(self):
        return self.vel_x, -self.vel_y

    @property
    def abs_vel(self):
        return math.sqrt(self.vel_x**2 + self.vel_y**2)

    @property
    def direction(self):
        return math.atan2(self.vel_y, self.vel_x)

    def step(self):
        if self.alive:
            self.x += self.vel_x * self.dt
            self.y += self.vel_y * self.dt

            self.vel_x = self.vel_x * self.decay
            self.vel_y = self.vel_y * self.decay + 0.5 * self.g * self.dt**2

        self.ticks += 1
        if self.ticks * self.dt > self.lifetime:
            self.alive = False

class Ball(Particle):
    def __init__(self, surface, color, radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color
        self.surface = surface
        self.radius = radius

    def draw(self):
        pygame.draw.circle(self.surface, self.color, (particle.x, -self.y), self.radius)



if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    fps = 60
    clock = pygame.time.Clock()
    ticks = 0

    w, h = screen.get_width(), screen.get_height()

    def random_ball():
        s = 1 if rand() > 0.5 else -1
        return Ball(screen, (randint(0,255), randint(0,255), randint(0,255)), 1, pos=(w//2, h*0.9), vel=((rand()-0.5)*w/5, rand()*h/5+h/5), dt=1/fps, lifetime=3, linear_factor=1000 )

    particles = []


    running = True

    while running:
        ticks += 1
        if ticks % randint(1,5) == 0:
            for _ in range(randint(1,10)):
                particles.append(random_ball())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((20, 20, 20))
        for particle in particles:
            particle.draw()
            particle.step()

        particles = [x for x in particles if x.alive]

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()