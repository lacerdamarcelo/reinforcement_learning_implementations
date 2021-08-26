# -*- coding: utf-8 -*-
import os
import gym
import time
import numpy as np
from tqdm import tqdm
from vacuum_robot_discrete_v0 import DiscreteVaccumRobotV0
"""
Implementation of Q-Learning with adaptive state clustering. for the Vacuum Robot Discrete V0 environment.
"""

def e_greedy_policy(q, current_state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(q[current_state]))
    else:
        return np.argmax(q[current_state])

def get_next_room_memory(room_memory, found_obstacle, next_position, next_rotation):
    new_room_memory = room_memory.copy()
    if found_obstacle:
        if next_rotation == 0:
            new_room_memory[next_position[0] - 1][next_position[1]] = 2
        elif next_rotation == 90:
            new_room_memory[next_position[0]][next_position[1] + 1] = 2
        elif next_rotation == 180:
            new_room_memory[next_position[0] + 1][next_position[1]] = 2
        else:
            new_room_memory[next_position[0]][next_position[1] - 1] = 2
    else:
        new_room_memory[next_position[0]][next_position[1]] = 1
    return new_room_memory

def infer_state(room_memory, position, rotation):
    # TO BE IMPLEMENTED
    return np.random.randint(0, 10)

def run(n_states, n_actions, max_episodes, env, gamma, epsilon, alpha):
    q = np.zeros((n_states, n_actions))
    for episode in range(max_episodes):
        terminal = False
        current_observables = env.reset()
        current_room_memory = np.zeros(env.room_data_shape)
        current_position = [current_observables[0], current_observables[1]]
        current_room_memory[current_position[0]][current_position[1]] = 1
        current_rotation = current_observables[2]
        current_state = infer_state(current_room_memory, current_position, current_rotation)
        print(current_room_memory)
        while terminal is False:
            action = e_greedy_policy(q, current_state, epsilon)
            next_observables, reward, terminal, info = env.step(action)
            print(current_position, current_rotation, action)
            next_position = [next_observables[0], next_observables[1]]
            next_rotation = next_observables[2]
            found_obstacle = next_observables[3]
            next_room_memory = get_next_room_memory(current_room_memory, found_obstacle, next_position, next_rotation)
            next_state = infer_state(next_room_memory, next_position, next_rotation)
            q[current_state][action] += alpha * (reward + gamma * np.max(q[next_state]) - \
                                        q[current_state][action])
            current_state = next_state
            current_position = next_position
            current_rotation = next_rotation
            current_room_memory = next_room_memory
            print(current_room_memory)
            time.sleep(1)
        print('================')
    return q

env = DiscreteVaccumRobotV0('room1.csv', 100)
q = run(10, 3, 100, env, 0.99, 0.1, 0.9)
