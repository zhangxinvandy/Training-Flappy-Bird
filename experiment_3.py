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
from collections import defaultdict

logging.basicConfig(
    # filename="log_exp3.txt",
    format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

ACTIONS = 2  # number of valid actions
GAMMA = 1  # decay rate of past observations
EXPLORE = 3000000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
ALPHA = 0.7  # learning rate
GRID = 10


def image_process(orig_img, t):
    img = cv2.cvtColor(cv2.resize(orig_img, (80, 80)), cv2.COLOR_BGR2GRAY)  # change color image to greyscale, after
    # resizing the image to be 80*80
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (80, 80, 1))
    # cv2.imwrite("x" + str(t) + ".png", img)

    # Find the lower right corner pixel of bird
    bird = None
    for j in range(0, 60):
        box = img[16:25, j:j + 4]
        score = np.sum(box == 255)
        up = img[15, j:j + 4]
        up_check = np.sum(up == 255)
        down = img[25, j:j + 4]
        down_check = np.sum(down == 255)
        left = img[16:25, j]
        left_check = np.sum(left == 255)

        if score > 20 and up_check == 0 and down_check == 0 and left_check > 0:
            bird = [24, j + 3]  # coordinate of lower right corner of bird
            break

    # Find the closest pipe's lower right corner point's coordinate
    points = []
    for i in range(24, 80):  # row # start from 24 since pipe passed are not important anymore
        for j in range(1, 75):  # col
            if i == 79:
                if img[i, j] == 255 and img[i, j - 1] == 0 and j < 63:
                    points.append([i, j])
            else:
                check_rows = img[(i - 10):i, j]
                check_cols = img[i, j:(j + 3)]
                if all(v == 255 for v in check_rows) and all(v == 255 for v in check_cols) \
                        and img[i - 2, j - 1] == 0 and img[i + 1, j] == 0:
                    points.append([i, j])

    if bird is None:
        if np.sum(img[16, 0:60] == 255) < 10:
            bird = [24, np.argmax(img[16]) + 1]
        else:
            bird = [24, np.argmax(img[24]) + 1]
    if len(points) == 0:
        return 0, int((80-24)/GRID)
    else:
        select = points[np.argmin(points, axis=0)[0]]

        # Vertical distance from lower pipe
        vd = select[1] - bird[1]

        # Horizontal distance from lower pipe
        hd = select[0] - bird[0]

        return int(vd / GRID), int(hd / GRID)


def policy(Q, state, epsilon):
    # choose an action epsilon greedily
    q = Q[state]
    A = np.zeros(ACTIONS)
    if random.random() <= epsilon:
        action_index = random.randrange(ACTIONS)
    else:
        action_index = np.argmax(q)
    A[action_index] = 1
    return A


def greedy(Q, state):
    # choose an action greedily
    q = Q[state]
    A = np.zeros(ACTIONS)
    action_index = np.argmax(q)
    A[action_index] = 1
    return A


def train_game():
    Q = defaultdict(lambda: np.zeros(ACTIONS))

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing
    A = np.zeros(ACTIONS)
    A[0] = 1  # do nothing
    img, r_0, terminal = game_state.frame_step(A)
    t = 0
    vd, hd = image_process(img, t)
    s = tuple([vd, hd])

    epsilon = INITIAL_EPSILON
    t = 0
    success = 0
    episode = 0
    death_punish = 1

    # start training
    while True:
        img, r, done = game_state.frame_step(A)
        vd, hd = image_process(img, t)
        if r == 1:
            success += 1
        elif r == -1:
            episode += 1
            if s[0] > 4:
                death_punish = s[0] * 100
                r = death_punish * r

        new_s = tuple([vd, hd])
        new_A = policy(Q, new_s, epsilon)
        a = np.argmax(A)

        # Update Q
        if done:
            reward = -1000
            episode += 1
        else:
            reward = 1

        Q[s][a] = Q[s][a] + ALPHA * (reward + GAMMA * np.max(Q[new_s]) - Q[s][a])

        # scale down epsilon
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        print('Current Step %d Episode %d Success %d Epsilon %.6f Action %d Reward %f vd %f hd %f Punish %f'
                     % (t, episode, success, epsilon, a, r, s[0], s[1], death_punish))

        s = new_s
        A = new_A
        t += 1


def play_game():
    train_game()


def main():
    play_game()


if __name__ == "__main__":
    main()
