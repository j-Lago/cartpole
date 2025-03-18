import math
import random
from random import random as rand, randint, uniform, choice
import pygame
from _collections import deque


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

class BallParticle(Particle):
    def __init__(self, surface, color, radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color
        self.surface = surface
        self.radius = radius

    def draw(self):
        if self.alive:
            pygame.draw.circle(self.surface, self.color, (self.x, -self.y), self.radius)


class TextParticle(Particle):
    def __init__(self, surface, color, text, font, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color
        self.surface = surface
        self.text = text
        self.font = font

    def draw(self):
        if self.alive:
            text = self.font.render(self.text, True, self.color)
            self.surface.blit(text, (self.x, -self.y))


class Particles():
    def __init__(self, maxlen: int | None = None):
        self.particles = deque(maxlen=maxlen)

    def append(self, particle: Particle):
        self.particles.append(particle)

    def __len__(self):
        return len(self.particles)

    def step(self):
        for particle in self.particles:
            particle.step()

    def draw(self):
        for particle in self.particles:
            particle.draw()

    def step_and_draw(self):
        self.step()
        self.draw()

    def garbage_collect(self):
        self.particles = [x for x in self.particles if x.alive]



def example(spawn_every_n_ticks = (1,2), particles_per_spawn = (1,2), lifetime = (1,2), garbage_collect_every_n_ticks=None, maxlen=None, particle_type=BallParticle):
    window_size = (900, 600)
    fps = 60

    pygame.init()
    if window_size is None:
        screen_info = pygame.display.Info()
        screen = pygame.display.set_mode((screen_info.current_w, screen_info.current_h), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(window_size)

    w, h = (screen.get_width(), screen.get_height())
    clock = pygame.time.Clock()

    font = pygame.font.SysFont('Consolas', 22)
    small_font = pygame.font.SysFont('Consolas', 18)

    def random_particle(particle_type):
        if particle_type == BallParticle:
            return BallParticle(screen, (randint(0,255), randint(0,255), randint(0,255)), randint(1,2), pos=(w//2, h*0.9), vel=((rand()-0.5)*w/4, rand()*h/4+h/5), dt=1/fps, lifetime=uniform(lifetime[0], lifetime[1]), linear_factor=1000 )
        elif particle_type == TextParticle:
            return TextParticle(screen, (randint(0, 255), randint(0, 255), randint(0, 255)), f'{randint(-100, 100):+d}', small_font, pos=(w // 2, h * 0.9), vel=((rand() - 0.5) * w / 4, rand() * h / 4 + h / 5), dt=1 / fps, lifetime=uniform(lifetime[0], lifetime[1]), linear_factor=1000)
        else:
            assert False
    particles = Particles(maxlen)

    ticks = 0
    running = True
    while running:
        ticks += 1
        if ticks % randint(*spawn_every_n_ticks) == 0:
            for _ in range(randint(*particles_per_spawn)):
                particles.append(random_particle(particle_type))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        # for particle in particles:
        #     particle.draw()
        #     particle.step()

        # particles.step()
        # particles.draw()
        particles.step_and_draw()

        if ticks % garbage_collect_every_n_ticks == 0:
            particles = [x for x in particles if x.alive]

        text_len = font.render(f"particles: {len(particles)}", True, (180, 150, 60))
        screen.blit(text_len, (20, 20))

        text_fps = font.render(f"fps: {clock.get_fps():.1f}", True, (180, 150, 60))
        screen.blit(text_fps, (20, 46))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == '__main__':
    # example(spawn_every_n_ticks=(1, 5), particles_per_spawn=(1, 200), lifetime=(1.0, 4.0))
    example(spawn_every_n_ticks=(1, 20), particles_per_spawn=(100, 500), lifetime=(2.0, 6.0), garbage_collect_every_n_ticks=1000, maxlen=15000, particle_type=TextParticle)