#!/usr/bin/env python3
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright (C) 2008  Nick Redshaw
#    Copyright (C) 2018  Francisco Sanchez Arroyo
#

# TODO
# safe area on new life
# sounds thump

# Notes:
# random.randrange returns an int
# random.uniform returns a float
# p for pause
# j for toggle showing FPS
# o for frame advance whilst paused
from PIL import Image
import numpy as np
import pygame
import sys
import os
import random
from pygame.locals import *
from util.vectorsprites import *
from ship import *
from stage import *
from badies import *
from shooter import *
from soundManager import *
from neural_network import carac_extract
from neural_network.perceptron import Perceptron
from pynput.keyboard import Key, Controller


class Asteroids():
    explodingTtl = 180

    def __init__(self):
        self.stage = Stage('Atari Asteroids', (1024, 768))
        self.paused = False
        self.showingFPS = False
        self.frameAdvance = False
        self.gameState = "attract_mode"
        self.rockList = []
        self.createRocks(3)
        self.saucer = None
        self.secondsCount = 1
        self.score = 0
        self.ship = None
        self.lives = 0
        self.gamemode = 'normal'

        self.current_inputs = [0, 0, 0, 0]

    def initialiseGame(self):
        self.gameState = 'playing'
        [self.stage.removeSprite(sprite)
         for sprite in self.rockList]  # clear old rocks
        if self.saucer is not None:
            self.killSaucer()
        self.startLives = 5
        self.createNewShip()
        self.createLivesList()
        self.score = 0
        self.rockList = []
        self.numRocks = 3
        self.nextLife = 10000

        self.createRocks(self.numRocks)
        self.secondsCount = 1

    def createNewShip(self):
        if self.ship:
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
        self.ship = Ship(self.stage)
        self.stage.addSprite(self.ship.thrustJet)
        self.stage.addSprite(self.ship)

    def createLivesList(self):
        self.lives += 1
        self.livesList = []
        for i in range(1, self.startLives):
            self.addLife(i)

    def addLife(self, lifeNumber):
        self.lives += 1
        ship = Ship(self.stage)
        self.stage.addSprite(ship)
        ship.position.x = self.stage.width - \
                          (lifeNumber * ship.boundingRect.width) - 10
        ship.position.y = 0 + ship.boundingRect.height
        self.livesList.append(ship)

    def createRocks(self, numRocks):
        for _ in range(0, numRocks):
            position = Vector2d(random.randrange(-10, 10),
                                random.randrange(-10, 10))

            newRock = Rock(self.stage, position, Rock.largeRockType)
            self.stage.addSprite(newRock)
            self.rockList.append(newRock)

    def pressInput(self, input):
        """
        Press the imputs in parameters
        input : array of the 4 keys (ex : [0 1 0 1] means right + shoot)
        """
        keyboard = Controller()
        if input[0] == 1:
            # self.pressKey(pygame.locals.K_LEFT)
            # print(type(key[K_LEFT]))
            keyboard.press(Key.left)
            # print("gauche")
        else:
            # self.releaseKey(pygame.locals.K_LEFT)
            # key[K_LEFT] = 0
            keyboard.release(Key.left)
        if input[1] == 1:
            keyboard.press(Key.right)
            # print("droite")
        else:
            keyboard.release(Key.right)
        # self.pressKey(pygame.locals.K_RIGHT)
        if input[2] == 1:
            keyboard.press(Key.up)
            # print("haut")
        else:
            keyboard.release(Key.up)
        # self.pressKey(pygame.locals.K_UP)
        if input[3] == 1:
            keyboard.press(Key.space)
            # print("tir")
        else:
            keyboard.release(Key.space)

    def playGame(self):

        # self.gamemode = 'automatic'  # or normal
        self.gamemode = 'normal'

        if self.gamemode == 'automatic':
            debug = False
            perceptron = Perceptron()
            perceptron.model()
            perceptron.load_model(
                "./neural_network/model 19-02-2020_01-13-26_78_acc_23_val.h5")
            perceptron.load_dataset(debug=debug)

        clock = pygame.time.Clock()

        i = 0

        frameCount = 0.0
        timePassed = 0.0
        self.fps = 0.0
        # Main loop
        while True:

            # calculate fps
            timePassed += clock.tick(60)
            frameCount += 1
            if frameCount % 10 == 0:  # every 10 frames
                # nearest integer
                self.fps = round((frameCount / (timePassed / 1000.0)))
                # reset counter
                timePassed = 0
                frameCount = 0

            self.secondsCount += 1

            if self.gamemode == 'automatic':
                frame_data = self.carac.get_dataframe(self.ship, self.rockList,
                                                      self.lives, self.score)
                if frame_data is not None and self.gameState == 'playing':
                    next_input = perceptron.predict(framedata=frame_data,
                                                    debug=debug)
                    print(next_input)
                    self.pressInput(
                        carac_extract.convert_to_simple_input(next_input))

            i += 1

            self.input(pygame.event.get())

            # pause
            if self.paused and not self.frameAdvance:
                self.displayPaused()
                continue

            self.stage.screen.fill((10, 10, 10))
            self.stage.moveSprites()
            self.stage.drawSprites()
            # self.doSaucerLogic()
            self.displayScore()
            if self.showingFPS:
                self.displayFps()  # for debug
            self.checkScore()

            # Process keys
            if self.gameState == 'playing':
                self.playing()

            elif self.gameState == 'exploding':
                self.exploding()
            else:
                self.displayText()

            # self.carac.display_rock_position(self.rockList)
            # self.carac.display_score(self.score)
            # self.carac.display_number_life(self.lives)
            # self.carac.display_ship(self.ship)
            # self.carac.get_data(self.ship,self.rockList,self.lives,self.score)

            # Double buffer draw
            pygame.display.flip()

    def get_screen_as_nparray(self):
        """To get image"""
        img = pygame.surfarray.array3d(self.stage.screen)
        return img.swapaxes(0, 1)

    def playing(self):
        if self.lives == 0:
            self.gameState = 'attract_mode'
        else:
            self.processKeys()
            self.checkCollisions()
            if len(self.rockList) == 0:
                self.levelUp()

    def doSaucerLogic(self):
        if self.saucer is not None:
            if self.saucer.laps >= 2:
                self.killSaucer()

        # Create a saucer
        if self.secondsCount % 2000 == 0 and self.saucer is None:
            randVal = random.randrange(0, 10)
            if randVal <= 3:
                self.saucer = Saucer(
                    self.stage, Saucer.smallSaucerType, self.ship)
            else:
                self.saucer = Saucer(
                    self.stage, Saucer.largeSaucerType, self.ship)
            self.stage.addSprite(self.saucer)

    def exploding(self):
        self.explodingCount += 1
        if self.explodingCount > self.explodingTtl:
            self.gameState = 'playing'
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
            self.ship.shipDebrisList = []

            if self.lives == 0:
                self.ship.visible = False
                self.gameState == "done"
            else:
                self.createNewShip()

    def is_done(self):
        return self.gameState == "done"

    def levelUp(self):
        self.numRocks += 1
        self.createRocks(self.numRocks)

    # move this kack somewhere else!
    def displayText(self):
        font1 = pygame.font.Font('../res/Hyperspace.otf', 50)
        font2 = pygame.font.Font('../res/Hyperspace.otf', 20)
        font3 = pygame.font.Font('../res/Hyperspace.otf', 30)

        titleText = font1.render('Asteroids', True, (180, 180, 180))
        titleTextRect = titleText.get_rect(centerx=self.stage.width / 2)
        titleTextRect.y = self.stage.height / 2 - titleTextRect.height * 2
        self.stage.screen.blit(titleText, titleTextRect)

        keysText = font2.render(
            '(C) 1979 Atari INC.', True, (255, 255, 255))
        keysTextRect = keysText.get_rect(centerx=self.stage.width / 2)
        keysTextRect.y = self.stage.height - keysTextRect.height - 20
        self.stage.screen.blit(keysText, keysTextRect)

        instructionText = font3.render(
            'Press start to Play', True, (200, 200, 200))
        instructionTextRect = instructionText.get_rect(
            centerx=self.stage.width / 2)
        instructionTextRect.y = self.stage.height / 2 - instructionTextRect.height
        self.stage.screen.blit(instructionText, instructionTextRect)

    def displayScore(self):
        font1 = pygame.font.Font('../res/Hyperspace.otf', 30)
        scoreStr = str("%02d" % self.score)
        scoreText = font1.render(scoreStr, True, (200, 200, 200))
        scoreTextRect = scoreText.get_rect(centerx=100, centery=45)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def displayPaused(self):
        if self.paused:
            font1 = pygame.font.Font('../res/Hyperspace.otf', 30)
            pausedText = font1.render("Paused", True, (255, 255, 255))
            textRect = pausedText.get_rect(
                centerx=self.stage.width / 2, centery=self.stage.height / 2)
            self.stage.screen.blit(pausedText, textRect)
            pygame.display.update()

    # Should move the ship controls into the ship class
    def input(self, events):
        self.frameAdvance = False
        if self.gameState == 'attract_mode':
            # Start a new game
            # if event.key == K_RETURN:
            print("lancement")
            self.initialiseGame()
        for event in events:
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    if self.gamemode == 'normal':
                        sys.exit(0)
                if event.key == K_s:
                    if self.gameState == 'playing':
                        if event.key == K_SPACE:
                            self.ship.fireBullet()
                            self.current_inputs[3] = 1
                            # print('Fire bullet')
                        elif event.key == K_b:
                            self.ship.fireBullet()
                            self.current_inputs[3] = 1
                        elif event.key == K_h:
                            self.ship.enterHyperSpace()
                            self.current_inputs[3] = 0
                        else:
                            self.current_inputs[3] = 0
                # elif self.gameState == 'attract_mode':
                #     # Start a new game
                #     if event.key == K_RETURN:
                #         self.initialiseGame()

                if event.key == K_p:
                    if self.paused:  # (is True)
                        self.paused = False
                    else:
                        self.paused = True

                if event.key == K_j:
                    if self.showingFPS:  # (is True)
                        self.showingFPS = False
                    else:
                        self.showingFPS = True

                if event.key == K_f:
                    pygame.display.toggle_fullscreen()

                # if event.key == K_k:
                # self.killShip()
            elif event.type == KEYUP:
                if event.key == K_o:
                    self.frameAdvance = True

    def processKeys(self):
        key = pygame.key.get_pressed()

        if key[K_LEFT] or key[K_z]:
            self.ship.rotateLeft()
            self.current_inputs[0] = 1
            self.current_inputs[1] = 0
            # print('left')
        elif key[K_RIGHT] or key[K_x]:
            self.ship.rotateRight()
            self.current_inputs[0] = 0
            self.current_inputs[1] = 1
            # print('right')

        if key[K_UP] or key[K_n]:
            self.ship.increaseThrust()
            self.ship.thrustJet.accelerating = True
            self.current_inputs[2] = 1
            # print('up')
        else:
            self.ship.thrustJet.accelerating = False
            self.current_inputs[2] = 0

    # Check for ship hitting the rocks etc.

    def checkCollisions(self):

        # Ship bullet hit rock?
        newRocks = []
        shipHit, saucerHit = False, False

        # Rocks
        for rock in self.rockList:
            rockHit = False

            if not self.ship.inHyperSpace and rock.collidesWith(self.ship):
                p = rock.checkPolygonCollision(self.ship)
                if p is not None:
                    shipHit = True
                    rockHit = True

            if self.saucer is not None:
                if rock.collidesWith(self.saucer):
                    saucerHit = True
                    rockHit = True

                if self.saucer.bulletCollision(rock):
                    rockHit = True

                if self.ship.bulletCollision(self.saucer):
                    saucerHit = True
                    self.score += self.saucer.scoreValue

            if self.ship.bulletCollision(rock):
                rockHit = True

            if rockHit:
                self.rockList.remove(rock)
                self.stage.spriteList.remove(rock)

                if rock.rockType == Rock.largeRockType:
                    playSound("explode1")
                    newRockType = Rock.mediumRockType
                    self.score += 50
                elif rock.rockType == Rock.mediumRockType:
                    playSound("explode2")
                    newRockType = Rock.smallRockType
                    self.score += 100
                else:
                    playSound("explode3")
                    self.score += 200

                if rock.rockType != Rock.smallRockType:
                    # new rocks
                    for _ in range(0, 2):
                        position = Vector2d(rock.position.x, rock.position.y)
                        newRock = Rock(self.stage, position, newRockType)
                        self.stage.addSprite(newRock)
                        self.rockList.append(newRock)

                self.createDebris(rock)

        # Saucer bullets
        if self.saucer is not None:
            if not self.ship.inHyperSpace:
                if self.saucer.bulletCollision(self.ship):
                    shipHit = True

                if self.saucer.collidesWith(self.ship):
                    shipHit = True
                    saucerHit = True

            if saucerHit:
                self.createDebris(self.saucer)
                self.killSaucer()

        if shipHit:
            self.killShip()

            # comment in to pause on collision
            # self.paused = True

    def killShip(self):
        stopSound("thrust")
        playSound("explode2")
        self.explodingCount = 0
        self.lives -= 1
        if (self.livesList):
            ship = self.livesList.pop()
            self.stage.removeSprite(ship)

        self.stage.removeSprite(self.ship)
        self.stage.removeSprite(self.ship.thrustJet)
        self.gameState = 'exploding'
        self.ship.explode()

    def killSaucer(self):
        stopSound("lsaucer")
        stopSound("ssaucer")
        playSound("explode2")
        self.stage.removeSprite(self.saucer)
        self.saucer = None

    def createDebris(self, sprite):
        for _ in range(0, 25):
            position = Vector2d(sprite.position.x, sprite.position.y)
            debris = Debris(position, self.stage)
            self.stage.addSprite(debris)

    def displayFps(self):
        font2 = pygame.font.Font('../res/Hyperspace.otf', 15)
        fpsStr = str(self.fps) + (' FPS')
        scoreText = font2.render(fpsStr, True, (255, 255, 255))
        scoreTextRect = scoreText.get_rect(
            centerx=(self.stage.width / 2), centery=15)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def checkScore(self):
        if self.score > 0 and self.score > self.nextLife:
            playSound("extralife")
            self.nextLife += 10000
            self.addLife(self.lives)


# Script to run the game
if not pygame.font:
    print('Warning, fonts disabled')
if not pygame.mixer:
    print('Warning, sound disabled')

initSoundManager()
game = Asteroids()  # create object game from class Asteroids
game.playGame()

####
