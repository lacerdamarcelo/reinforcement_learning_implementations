# -*- coding: utf-8 -*-

import os
import gym
import time
import numpy as np
from tqdm import tqdm
import plotly.express as px
from scipy.spatial.distance import cdist
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

def calculate_distance_between_observations(vec_a, vec_b):
    return (vec_a != vec_b).sum()

def infer_state(room_memory, position, rotation, window_size, state_counters, state_threshold, q, n_actions):
    surroundings = np.full((window_size, window_size), 2)
    row_min = position[0] - int(window_size / 2)
    row_max = position[0] + int(window_size / 2) + 1
    column_min = position[1] - int(window_size / 2)
    column_max = position[1] + int(window_size / 2) + 1
    selection = room_memory[row_min if row_min >= 0 else 0: row_max if row_max < room_memory.shape[0] else room_memory.shape[0],
                            column_min if column_min >= 0 else 0: column_max if column_max < room_memory.shape[1] else room_memory.shape[1]]
    row_offset = 0
    column_offset = 0
    if row_min < 0:
        row_offset = -row_min
    if column_min < 0:
        column_offset = -column_min

    surroundings[row_offset: selection.shape[0] + row_offset, column_offset: selection.shape[1] + column_offset] = selection
    if rotation == 90:
        surroundings = np.rot90(surroundings)
    elif rotation == 180:
        surroundings = np.rot90(surroundings)
        surroundings = np.rot90(surroundings)
    elif rotation == 270:
        surroundings = np.rot90(surroundings)
        surroundings = np.rot90(surroundings)
        surroundings = np.rot90(surroundings)
    surroundings = surroundings.flatten().astype(int)
    if len(state_counters) == 0:
        new_counters = []
        for value in surroundings:
            new_values = np.array([0, 0, 0])
            new_values[value] += 1
            new_counters.append(new_values)
        state_counters.append(new_counters)
        q = np.zeros((1, n_actions))
        return 0, q
    else:
        distances = []
        for counters in state_counters:
            state_vector = [counter.argmax() for counter in counters]
            distance = calculate_distance_between_observations(state_vector, surroundings)
            distances.append(distance)
        min_distance = np.min(distances)
        min_distance_state = np.argmin(distances)
        #print(min_distance, round(state_threshold * len(surroundings)))
        if min_distance <= round(state_threshold * len(surroundings)):
            for i, value in enumerate(surroundings):
                state_counters[min_distance_state][i][value] += 1
            return min_distance_state, q
        else:
            new_counters = []
            for value in surroundings:
                new_values = np.array([0, 0, 0])
                new_values[value] += 1
                new_counters.append(new_values)
            state_counters.append(new_counters)
            q = np.append(q, np.array([q[min_distance_state].copy()]), axis=0)
            return len(state_counters) - 1, q

def run(n_actions, max_episodes, env, gamma, epsilon, alpha, window_size, state_threshold):
    q = np.zeros((0, n_actions))
    state_counters = []
    rewards_sums = []
    for episode in range(max_episodes):
        terminal = False
        current_observables = env.reset()
        current_room_memory = np.zeros(env.room_data_shape)
        current_position = [current_observables[0], current_observables[1]]
        current_room_memory[current_position[0]][current_position[1]] = 1
        current_rotation = current_observables[2]
        current_state, q = infer_state(current_room_memory, current_position, current_rotation, window_size, state_counters,
                                       state_threshold, q, n_actions)
        rewards_sum = 0
        while terminal is False:
            action = e_greedy_policy(q, current_state, epsilon)
            next_observables, reward, terminal, info = env.step(action)
            next_position = [next_observables[0], next_observables[1]]
            next_rotation = next_observables[2]
            found_obstacle = next_observables[3]
            next_room_memory = get_next_room_memory(current_room_memory, found_obstacle, next_position, next_rotation)
            next_state, q = infer_state(next_room_memory, next_position, next_rotation, window_size, state_counters,
                                        state_threshold, q, n_actions)
            #print(next_state, q)
            q[current_state][action] += alpha * (reward + gamma * np.max(q[next_state]) - \
                                        q[current_state][action])
            current_state = next_state
            current_position = next_position
            current_rotation = next_rotation
            current_room_memory = next_room_memory
            
            '''
            for counters in state_counters:
                for c in counters:
                    print(c.argmax(), end=', ')
                print('')
            print('=======')
            time.sleep(0.1)
            '''
            
            rewards_sum += reward
            #print(state_counters)
            #print(current_position, current_rotation)
            #print(current_room_memory)
            #env.render()
            #time.sleep(0.5)
        #print(current_room_memory)
        print(episode, rewards_sum)
        rewards_sums.append(rewards_sum)
    #for counters in state_counters:
    #    for c in counters:
    #        print(c.argmax(), end=', ')
    #    print('')
    #print('=======')
    return q, rewards_sums

env = DiscreteVaccumRobotV0('room1.csv', 25)
q, rewards_sum = run(3, 10000, env, 0.99, 0.1, 0.9, 5, 0.0)
fig = px.line(x=range(len(rewards_sum)), y=rewards_sum)
fig.show()