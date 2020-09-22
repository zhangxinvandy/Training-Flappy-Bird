#!/usr/bin/env python
from __future__ import print_function
import os

os.environ['TF_KERAS'] = '1'

import sys

sys.path.append("game/")
import wrapped_flappy_bird_no_image as game
import random
import logging
import pickle
import numpy as np
from collections import defaultdict

logging.basicConfig(
    # filename="log_exp5.txt",
    format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

ACTIONS = 2  # number of valid actions
GAMMA = 1  # decay rate of past observations
EXPLORE = 3000000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
ALPHA = 0.7  # learning rate


def policy(Q, state, epsilon):
    if random.random() <= epsilon:
        action_index = random.randrange(ACTIONS)
    else:
        action_index = np.argmax(Q[state])
    return action_index


def map_state(xdif, ydif, vel):
    if xdif < 140:
        xdif = int(xdif) - (int(xdif) % 10)
    else:
        xdif = int(xdif) - (int(xdif) % 70)

    if ydif < 180:
        ydif = int(ydif) - (int(ydif) % 10)
    else:
        ydif = int(ydif) - (int(ydif) % 60)

    return str(int(xdif)) + "_" + str(int(ydif)) + "_" + str(vel)


def train_game():
    Q = defaultdict(lambda: np.zeros(ACTIONS))
    t = 0
    score = 0
    success = 0
    episode = 0
    epsilon = INITIAL_EPSILON

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    img, r, done = game_state.frame_step(np.array([1, 0]))
    playery = game_state.playery
    playerx = game_state.playerx
    if -playerx + game_state.lowerPipes[0]["x"] > -30:
        myPipe = game_state.lowerPipes[0]
    else:
        myPipe = game_state.lowerPipes[1]

    xdif = -playerx + myPipe["x"]
    ydif = -playery + myPipe["y"]
    vel = game_state.playerVelY
    s = map_state(xdif, ydif, vel)
    a = policy(Q, s, epsilon)

    # start training
    while True:
        A = np.zeros(ACTIONS)
        A[a] = 1
        img, r, done = game_state.frame_step(A)
        if r == 1:
            success += 1
            score += 1
        playery = game_state.playery
        playerx = game_state.playerx
        if -playerx + game_state.lowerPipes[0]["x"] > -30:
            myPipe = game_state.lowerPipes[0]
        else:
            myPipe = game_state.lowerPipes[1]

        xdif = -playerx + myPipe["x"]
        ydif = -playery + myPipe["y"]
        vel = game_state.playerVelY
        new_s = map_state(xdif, ydif, vel)
        new_a = policy(Q, new_s, epsilon)

        # Update Q
        if done:
            reward = -1000
        else:
            reward = 1

        Q[s][a] = Q[s][a] + ALPHA * (reward + GAMMA * np.max(Q[new_s]) - Q[s][a])

        s = new_s
        a = new_a
        t += 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if done:
            print('Current Step %d Episode %d Score %d Success %d Action %d Reward %f state %s'
                  % (t, episode, score, success, a, r, s))

            score = 0
            episode += 1


def play_game():
    train_game()


def main():
    play_game()


if __name__ == "__main__":
    main()
