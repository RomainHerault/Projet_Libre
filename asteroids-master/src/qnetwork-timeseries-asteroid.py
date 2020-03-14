#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, LSTM, TimeDistributed
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K

import pickle
import os

import numpy as np
import random
from environement import *

# from keras.utils.training_utils import multi_gpu_model

EPISODES = 50000


# DRQN Agent at Breakout
class DRQNAgent:

    def __init__(self, action_size):
        self.render = False
        self.load_model = True
        # Define size of behavior
        self.action_size = action_size

        # DRQN Hyperparameters
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99

        # Replay memory, max size 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30

        # Create model and target model, initialize target model and assign model to gpu

        self.model = self.build_model()
        # self.model = multi_gpu_model(self.model, gpus = 4)
        self.target_model = self.build_model()
        # self.target_model = multi_gpu_model(self.target_model, gpus = 4)

        self.update_target_model()
        self.optimizer = self.optimizer()

        # Tensor Board Settings
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_drqn15', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # Define previous param if model if loaded
        self.prev_EPISODES = 0
        self.prev_global_step = 0

        if self.load_model:
            self.model.load_weights("save_model/asteroids_drqn15.h5")
            self.model.load_weights("save_model/asteroids_drqn15_target.h5")

            pickle_in = open("save_model/memory.pickle", "rb")
            self.memory = pickle.load(pickle_in)
            pickle_in.close()

            load_training_param(self)

    # Store samples <s, a, r, s'> in replay memory
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # Directly define optimization function to use Huber Loss
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')
        prediction = self.model.output
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)
        return train

    # Create a neural network with state as input and queue function output
    def build_model(self):
        model = Sequential()
        model.add(TimeDistributed(
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            input_shape=(10, 84, 84, 1)))

        # input_shape=(time_step, row, col, channels)
        model.add(TimeDistributed(
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu')))
        model.add(TimeDistributed(
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(512))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # Update the target model with the weight of the model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Choosing behavior with the epsilon greed policy
    def get_action(self, history):
        # modified
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # Train models with randomly extracted batches from replay memory
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)
        history = np.zeros((self.batch_size, 10, 84, 84, 1))
        next_history = np.zeros((self.batch_size, 10, 84, 84, 1))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])
        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * np.amax(
                    target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # Record learning information for each episode
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average_Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# Preprocessing with black and white screen to speed up learning
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


def save_training_param(agent, e, global_step):
    pickle_out = open("save_model/training_param.pickle", "wb")
    pickle.dump([e, global_step, agent.epsilon, agent.avg_q_max, agent.avg_loss], pickle_out)
    pickle_out.close()

def load_training_param(agent):
    pickle_in = open("save_model/training_param.pickle", "rb")
    tupl = pickle.load(pickle_in)
    pickle_in.close()

    agent.prev_EPISODES = tupl[0]
    agent.prev_global_step = tupl[1]
    agent.epsilon = tupl[2]
    agent.avg_q_max = tupl[3]
    agent.avg_loss = tupl[4]





if __name__ == "__main__":

    # create save_model folder
    if not os.path.isdir("save_model"):
        os.mkdir("save_model")

    # Your environment and DRQN ​​agents
    env = Environement()
    agent = DRQNAgent(action_size=12)
    scores, episodes, global_step = [], [], agent.prev_global_step
    for e in range(agent.prev_EPISODES,EPISODES):
        done = False
        dead = False
        step, score, start_life = 0, 0, 5
        observe = env.reset()
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)
        state = pre_processing(observe)
        state = state.reshape(84, 84, 1)
        history = np.stack((state, state, state, state, state,
                            state, state, state, state, state), axis=0)
        # ( 10, 84, 84, 1 )
        history = np.reshape([history], (1, 10, 84, 84, 1))

        while not done:
            # if agent.render:
            #     env.render()
            global_step += 1
            step += 1

            # Choose your action from the previous 4 states
            action = agent.get_action(history)

            # 1: stop, 2: left, 3: right
            # if action == 0:
            #     real_action = 1
            # elif action == 1:
            #     real_action = 2
            # else:
            #     real_action = 3

            # One time step in the environment with the selected action
            observe, reward, done, info = env.step(action)
            # reward = reward * 10
            # State preprocessing for each time step
            next_state = pre_processing(observe)
            next_state = next_state.reshape(1, 84, 84, 1)
            next_history = next_state.reshape(1, 1, 84, 84, 1)
            next_history = np.append(next_history, history[:, :9, :, :, :],
                                     axis=1)
            next_history = np.reshape([next_history], (1, 10, 84, 84, 1))
            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])
            if start_life > info:
                dead = True
                start_life = info
            reward = np.clip(reward, -1., 1.)

            # Save sample <s, a, r, s'>
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # Update the target model with the weight of the model every time
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history
            if done:
                # Record learning information for each episode
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))
                agent.avg_q_max, agent.avg_loss = 0, 0

        # Save Model Every 1 Episodes
        if e % 1 == 0:
            agent.model.save_weights("save_model/asteroids_drqn15.h5")
            agent.model.save_weights("save_model/asteroids_drqn15_target.h5")

            pickle_out = open("save_model/memory.pickle", "wb")
            pickle.dump(agent.memory, pickle_out)
            pickle_out.close()

            save_training_param(agent, e, global_step)
