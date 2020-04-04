from asteroids import *
from soundManager import *
import pygame
import time


class Environement():
    """
    This class is the enviroement for the network.
    It's used to run the game Asteroid step by step
    and get all parameters <screenshot, reward, gameover, lives remaining>
    """

    def __init__(self):
        """
        Init the game.
        """

        self.prev_lives = 5
        self.prev_score = 0

        # Script to run the game
        if not pygame.font:
            print('Warning, fonts disabled')
        if not pygame.mixer:
            print('Warning, sound disabled')

        initSoundManager()  # can be commented to deactivate sound
        self.game = Asteroids()  # create object game from class Asteroids

    def reset(self):
        """
        Restart the game after a game over.
        :return:
        """
        # Script to run the game
        if not pygame.font:
            print('Warning, fonts disabled')
        if not pygame.mixer:
            print('Warning, sound disabled')

        initSoundManager()
        self.game = Asteroids()  # create object game from class Asteroids
        self.prev_score = 0
        self.prev_lives = 5

        return self.game.get_screen_as_nparray(), 0, self.game.is_done(), self.game.lives

    def step(self, action):
        """
        Execute one step of the game.
        Here we choose to skip 18 frames to gain time for training.
        :param action: the action to do ex: Key_UP,KEY_DOWN,FIRE...
        :return: <screenshot, reward, gameover, lives remaining>
        """
        for _ in range(18):  # skip 18 frames
            self.game.step(action)

        if self.game.lives < self.prev_lives:
            # if the network loose a life, reward is negative
            reward = -1
        else:
            # if the network get a better score than the previous one,
            # it means he has destroyed asteroid and get positive reward
            score_diff = self.game.score - self.prev_score
            reward = score_diff / 200
        self.prev_score = self.game.score
        self.prev_lives = self.game.lives

        return self.game.get_screen_as_nparray(), reward, self.game.is_done(), self.game.lives


if __name__ == '__main__':
    env = Environement()

    time.sleep(5)

    env.reset()
