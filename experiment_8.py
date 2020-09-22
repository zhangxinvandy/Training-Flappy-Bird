#!/usr/bin/env python
from __future__ import print_function
import os

os.environ['TF_KERAS'] = '1'

import tensorflow as tf
import cv2
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import logging
import pickle
import numpy as np
from collections import deque

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

logging.basicConfig(
    # filename="log_exp8.txt",
    format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 10000  # timesteps to observe before training
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def create_network():
    action_size = 2
    # Neural Net for Deep-Q learning Model
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4),
                            padding='SAME', activation='relu', input_shape=(80, 80, 4),
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01),
                            bias_initializer=tf.constant_initializer(0.01)))  # single image size: (80, 80, 1)
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    model.add(layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                            padding='SAME', activation='relu',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01),
                            bias_initializer=tf.constant_initializer(0.01)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                            padding='SAME', activation='relu',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01),
                            bias_initializer=tf.constant_initializer(0.01)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', bias_initializer=tf.constant_initializer(0.01)))
    model.add(layers.Dense(action_size))

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(lr=1e-6))

    return model


def train_network(model):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    x_t = tf.cast(x_t, tf.float32)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Include the step in the file name (uses `str.format`)
    checkpoint_dir = "saved_models/dqn/experiment_8"
    checkpoint_path = os.path.join(checkpoint_dir, "dqn-{step:09d}.ckpt")

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    if latest:
        # Load the previously saved weights
        model.load_weights(latest)

        glb_vars_dict = pickle_load(os.path.join(checkpoint_dir, 'glb_vars_dict.pkl'))

        try:
            t, success, epsilon, D = glb_vars_dict[latest]
        except KeyError:
            t, success, epsilon, D = glb_vars_dict[latest.replace("dqn/experiment_8", "dqn")]


    else:
        glb_vars_dict = {}
        t = 0
        success = 0

    # start training
    while True:
        # choose an action greedily
        q = model.predict(np.array([s_t]))
        a = np.zeros(ACTIONS)
        action_index = np.argmax(q)
        a[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a)
        if r_t == 1:
            success += 1
        # change color image to greyscale, after resizing the image to be 80*80
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        # bird, pipes, ground is white, background is black
        _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        # shape of state image (80, 80, 1)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        x_t1 = tf.cast(x_t1, tf.float32)
        # stack the latest 4 images
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            state_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            new_state_batch = [d[3] for d in minibatch]
            terminal_batch = [d[4] for d in minibatch]

            targets = model.predict(np.array(state_batch))
            q_sa = model.predict(np.array(new_state_batch))
            targets[range(BATCH), action_batch] = reward_batch + GAMMA * np.max(q_sa, axis=1) * np.invert(
                terminal_batch)

            model.train_on_batch(np.array(state_batch), targets)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            # Save the weights using the `checkpoint_path` format
            model.save_weights(checkpoint_path.format(step=t))
            glb_vars_dict[checkpoint_path.format(step=t)] = [t, success, epsilon, D]
            # Saving the objects:
            pickle_dump(glb_vars_dict, os.path.join(checkpoint_dir, 'glb_vars_dict.pkl'))

        # print info
        if t <= OBSERVE:
            period = "observe"
        else:
            period = "train"

        logging.info('Current Step %d Success %d State %s Epsilon %.6f Action %d Reward %f Q_MAX %f'
                     % (t, success, period, epsilon, action_index, r_t, np.max(q)))


def play_game():
    model = create_network()
    train_network(model)


def main():
    play_game()


if __name__ == "__main__":
    main()
