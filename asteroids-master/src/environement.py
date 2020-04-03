from asteroids import *
from soundManager import *
import pygame
import time


class Environement():
    def __init__(self):

        self.prev_lives = 0
        self.prev_score = 0
        # Script to run the game
        if not pygame.font:
            print('Warning, fonts disabled')
        if not pygame.mixer:
            print('Warning, sound disabled')

        initSoundManager()
        self.game = Asteroids()  # create object game from class Asteroids

    def reset(self):
        # Script to run the game
        if not pygame.font:
            print('Warning, fonts disabled')
        if not pygame.mixer:
            print('Warning, sound disabled')

        initSoundManager()
        self.game = Asteroids()  # create object game from class Asteroids
        self.prev_score = self.game.score
        self.prev_lives = self.game.lives

        return self.game.get_screen_as_nparray(), 0, self.game.is_done(), self.game.lives

    def play_game(self):
        self.game.playGame()

    def step(self, action):
        for _ in range(18):
            self.game.step(action)

        if self.game.lives < self.prev_lives:
            reward = -1
        else:
            score_diff = self.game.score - self.prev_score
            reward = score_diff / 200

        return self.game.get_screen_as_nparray(), reward, self.game.is_done(), self.game.lives

if __name__ == '__main__':
    env = Environement()

    time.sleep(5)

    env.reset()
