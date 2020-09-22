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
    # filename="log_exp7.txt",
    format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


ACTIONS = 2  # number of valid actions
GAMMA = 1  # decay rate of past observations
ALPHA = 0.7  # learning rate


# greedy policy
def policy(Q, state):
    if Q[state][0] >= Q[state][1]:
        return 0
    else:
        return 1


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
    a = policy(Q, s)

    t = 0
    score = 0
    success = 0
    episode = 0
    moves = []

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
        new_a = policy(Q, new_s)
        moves.append((s, a, new_s))

        if done:
            episode += 1
            history = list(reversed(moves))

            # Flag if the bird died in the top pipe
            high_death_flag = True if int(history[0][2].split("_")[1]) > 120 else False

            # Q-learning score updates
            final = 1
            for exp in history:
                state = exp[0]
                act = exp[1]
                res_state = exp[2]

                # Select reward
                if final == 1 or final == 2:
                    cur_reward = -1000
                elif high_death_flag and act:
                    cur_reward = -1000
                    high_death_flag = False
                else:
                    cur_reward = 1

                # Update
                Q[state][act] = (1-ALPHA) * (Q[state][act]) + ALPHA * (cur_reward + GAMMA*max(Q[res_state]))
                final += 1
            moves = []

            print('Current Step %d Episode %d Score %d Success %d Action %d Reward %f state %s'
                  % (t, episode, score, success, a, r, s))
            score = 0

        s = new_s
        a = new_a
        t += 1


def play_game():
    train_game()


def main():
    play_game()


if __name__ == "__main__":
    main()
