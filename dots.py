import pygame
from pygame.locals import *
import random, math
import numpy as np

SCREEN_SIZE = np.array([800, 600])
GOAL = np.array([int(SCREEN_SIZE[0]/2), 10])


class Dot(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((0, 0), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(SCREEN_SIZE[0]/2, SCREEN_SIZE[1] - 10))

        self.brain = Brain(1000)
        self.pos = np.array([self.rect.x, self.rect.y])
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])

        self.dead = False
        self.reached_goal = False
        self.is_best = False
        self.fitness = 0

    def draw(self, surface):
        if self.is_best:
            pygame.draw.circle(surface, (0, 255, 0), (self.rect.x - 1, self.rect.y - 1), 4)
        else:
            pygame.draw.circle(surface, (0, 0, 0), (self.rect.x - 1, self.rect.y - 1), 2)

    def update(self):
        self.check_dead()
        if not self.dead and not self.reached_goal:
            self.acc = self.brain.directions[self.brain.step]
            self.brain.step += 1
            self.vel = self.vel + self.acc
            self.vel[self.vel > 2] = 2
            self.vel[self.vel < -2] = -2
            self.pos = self.pos + self.vel
            self.rect.x, self.rect.y = tuple(self.pos)

    def check_dead(self):
        if ((self.pos - GOAL)**2).sum() <= 6:
            self.reached_goal = True
        if self.pos.any() < 1 or self.pos[0] > SCREEN_SIZE[0] or self.pos[1] > SCREEN_SIZE[1]:
            self.dead = True
        if self.brain.step >= len(self.brain.directions):
            self.dead = True

    def calculate_fitness(self):
        if self.reached_goal:
            self.fitness = 1/16 + 10000 / self.brain.step**2
        else:
            self.fitness = 1 / ((self.pos - GOAL)**2).sum()

    def get_child(self):
        child = Dot()
        child.brain.directions = self.brain.directions.copy()
        if not self.is_best:
            child.brain.mutate()
        return child


class Brain:
    def __init__(self, size):
        self.directions = []
        for i in range(size):
            self.directions.append(self.get_direction())
        self.step = 0

    def get_direction(self):
        angle = random.random()*2*math.pi
        return np.array([math.cos(angle), math.sin(angle)])

    def mutate(self):
        rate = .005
        for i in range(len(self.directions)):
            if random.random() < rate:
                self.directions[i] = self.get_direction()


class Population:
    def __init__(self, size):
        self.dots = []
        self.fitness_sum = 0
        self.min_step = 1000

        for i in range(size):
            self.dots.append(Dot())

    def draw(self, surface):
        for i in range(1, len(self.dots)):
            self.dots[i].draw(surface)
        self.dots[0].draw(surface)

    def update(self):
        for dot in self.dots:
            if dot.brain.step > self.min_step:
                dot.dead = True
            else:
                dot.update()

    def all_dots_dead(self):
        for dot in self.dots:
            if not dot.dead and not dot.reached_goal:
                return False
        return True

    def repopulate(self):
        self.calculate_fitness()
        self.fitness_sum = sum([i.fitness for i in self.dots])

        best_dot = self.set_best()
        print(best_dot.reached_goal)
        new_pop = [best_dot.get_child()]
        new_pop[0].is_best = True
        self.min_step = best_dot.brain.step

        for i in range(len(self.dots)):
            parent = self.select_parent()
            new_pop.append(parent.get_child())
        self.dots = new_pop.copy()
        return best_dot.brain.step

    def calculate_fitness(self):
        for dot in self.dots:
            dot.calculate_fitness()

    def set_best(self):
        best = self.dots[0]
        for dot in self.dots:
            if dot.fitness > best.fitness:
                best = dot
        return best

    def select_parent(self):
        child = random.random()*self.fitness_sum
        tally = 0
        for dot in self.dots:
            tally += dot.fitness
            if tally > child:
                return dot
        return print('Should not get here')


class App:
    def __init__(self):
        self.population = Population(100)
        self.screen = None
        self.font = None

        self.generation = 1
        self.step = 0
        self.fitness = 0

        self.on_init()

        self.is_running = True
        while self.is_running:
            self.on_loop()
        self.cleanup()

    def on_init(self):
        pygame.init()
        self.font = pygame.font.Font('CalibriL.ttf', 12)
        pygame.display.set_caption("Dots ML")
        self.screen = pygame.display.set_mode(tuple(SCREEN_SIZE))

    def on_loop(self):
        if not self.population.all_dots_dead():
            event = pygame.event.poll()
            self.check_quit(event)

            self.update()
            self.render()
        else:
            self.generation += 1
            self.step = self.population.repopulate()
            self.fitness = self.population.fitness_sum

    def check_quit(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                self.is_running = False
        elif event.type == pygame.QUIT:
            self.is_running = False

    def cleanup(self):
        pygame.quit()

    def update(self):
        self.population.update()

    def render(self):
        self.screen.fill((255, 255, 255))
        pygame.draw.circle(self.screen, (255, 0, 0), (GOAL[0] - 2, GOAL[1] - 2), 4)
        self.population.draw(self.screen)
        gen_text = self.font.render(f'Generation: {self.generation}', True, (0, 0, 0))
        step_text = self.font.render(f'Steps: {self.step}', True, (0, 0, 0))
        fitness_text = self.font.render(f'Fitness: {self.fitness}', True, (0, 0, 0))
        self.screen.blit(gen_text, (5, 0))
        self.screen.blit(step_text, (5, 10))
        self.screen.blit(fitness_text, (5, 20))

        pygame.display.flip()


if __name__ == '__main__':
    App()
