from asteroids import *
from soundManager import *
import pygame
import time


class Environement():
    def __init__(self):

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

        return self.game.get_screen_as_nparray(), self.game.score, self.game.gameState == 'done', self.game.lives

    def play_game(self):
        self.game.playGame()

    def step(self, action):

        self.game.step(action)

        return self.game.get_screen_as_nparray(), self.game.score, self.game.gameState == 'done', self.game.lives


if __name__ == '__main__':
    env = Environement()

    time.sleep(5)

    env.reset()
