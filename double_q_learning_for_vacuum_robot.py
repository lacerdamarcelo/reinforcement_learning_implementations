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
Implementation of Double Q-Learning with adaptive state clustering. for the Vacuum Robot Discrete V0 environment.
DOES NOT WORK WELL. STATE INFERENCE IS DUMB.
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

def infer_state(room_memory, position, rotation, window_size, state_counters, state_threshold, q1, q2, n_actions):
    new_state_counters = state_counters.copy()
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
    if len(new_state_counters) == 0:
        new_counters = []
        for value in surroundings:
            new_values = np.array([0, 0, 0])
            new_values[value] += 1
            new_counters.append(new_values)
        new_state_counters.append(new_counters)
        q1 = np.zeros((1, n_actions))
        q2 = np.zeros((1, n_actions))
        return 0, q1, q2, new_state_counters
    else:
        distances = []
        for counters in new_state_counters:
            state_vector = [counter.argmax() for counter in counters]
            distance = calculate_distance_between_observations(state_vector, surroundings)
            distances.append(distance)
        min_distance = np.min(distances)
        min_distance_state = np.argmin(distances)
        #print(min_distance, round(state_threshold * len(surroundings)))
        if min_distance <= round(state_threshold * len(surroundings)):
            for i, value in enumerate(surroundings):
                new_state_counters[min_distance_state][i][value] += 1
            return min_distance_state, q1, q2, new_state_counters
        else:
            new_counters = []
            for value in surroundings:
                new_values = np.array([0, 0, 0])
                new_values[value] += 1
                new_counters.append(new_values)
            new_state_counters.append(new_counters)
            q1 = np.append(q1, np.array([q1[min_distance_state].copy()]), axis=0)
            q2 = np.append(q2, np.array([q2[min_distance_state].copy()]), axis=0)
            return len(new_state_counters) - 1, q1, q2, new_state_counters

def run(n_actions, max_episodes, env, gamma, epsilon, alpha, window_size, state_threshold, num_test_rounds, policy_rollback_prob):
    q1 = np.zeros((0, n_actions))
    q2 = np.zeros((0, n_actions))
    state_counters = []
    rewards_avgs = []
    previous_reward_avg_test = None
    previous_q1 = None
    previous_q2 = None
    previous_state_counters = None
    for episode in range(max_episodes):
        terminal = False
        current_observables = env.reset()
        current_room_memory = np.zeros(env.room_data_shape)
        current_position = [current_observables[0], current_observables[1]]
        current_room_memory[current_position[0]][current_position[1]] = 1
        current_rotation = current_observables[2]

        current_state, q1, q2, state_counters = infer_state(current_room_memory, current_position, current_rotation, window_size,
                                                            state_counters, state_threshold, q1, q2, n_actions)
        while terminal is False:
            action = e_greedy_policy(q1 + q2, current_state, epsilon)
            next_observables, reward, terminal, info = env.step(action)
            next_position = [next_observables[0], next_observables[1]]
            next_rotation = next_observables[2]
            found_obstacle = next_observables[3]
            next_room_memory = get_next_room_memory(current_room_memory, found_obstacle, next_position, next_rotation)
            next_state, q1, q2, state_counters = infer_state(next_room_memory, next_position, next_rotation, window_size,
                                                             state_counters, state_threshold, q1, q2, n_actions)
            if np.random.random() > 0.5:
                q1[current_state][action] += alpha * (reward + gamma * q2[next_state][np.argmax(q1[next_state])] - \
                                             q1[current_state][action])
            else:
                q2[current_state][action] += alpha * (reward + gamma * q1[next_state][np.argmax(q2[next_state])] - \
                                             q2[current_state][action])
            current_state = next_state
            current_position = next_position
            current_rotation = next_rotation
            current_room_memory = next_room_memory

        rewards_sum = 0
        for test_round in range(num_test_rounds):
            terminal = False
            current_observables = env.reset()
            current_room_memory = np.zeros(env.room_data_shape)
            current_position = [current_observables[0], current_observables[1]]
            current_room_memory[current_position[0]][current_position[1]] = 1
            current_rotation = current_observables[2]
            current_state, q1, q2, _ = infer_state(current_room_memory, current_position, current_rotation,
                                                                window_size, state_counters, state_threshold, q1, q2, n_actions)
            while terminal is False:
                action = e_greedy_policy(q1 + q2, current_state, epsilon)
                next_observables, reward, terminal, info = env.step(action)
                rewards_sum += reward
        rewards_avg = rewards_sum / num_test_rounds

        if previous_reward_avg_test is None or rewards_avg > previous_reward_avg_test:
            previous_reward_avg_test = rewards_avg
            previous_q1 = q1.copy()
            previous_q2 = q2.copy()
            previous_state_counters = state_counters.copy()
        else:
            if np.random.random() < policy_rollback_prob:
                q1 = previous_q1
                q2 = previous_q2
                state_counters = previous_state_counters
            else:
                previous_reward_avg_test = rewards_avg
                previous_q1 = q1.copy()
                previous_q2 = q2.copy()
                previous_state_counters = state_counters.copy()

        print(episode, previous_reward_avg_test)
        rewards_avgs.append(previous_reward_avg_test)
    #for counters in state_counters:
    #    for c in counters:
    #        print(c.argmax(), end=', ')
    #    print('')
    #print('=======')
    return q1, q2, state_counters, rewards_avgs

env = DiscreteVaccumRobotV0('room1.csv', 50)
q1, q2, state_counters, rewards_sum = run(3, 200, env, 0.99, 0.1, 0.9, 3, 1/9, 30, 1.0)
fig = px.line(x=range(len(rewards_sum)), y=rewards_sum)
fig.show()

for test_round in range(30):
    terminal = False
    current_observables = env.reset()
    current_room_memory = np.zeros(env.room_data_shape)
    current_position = [current_observables[0], current_observables[1]]
    current_room_memory[current_position[0]][current_position[1]] = 1
    current_rotation = current_observables[2]
    current_state, _, _, _ = infer_state(current_room_memory, current_position, current_rotation,
                                                        3, state_counters, 0.0, q1, q2, 3)
    rewards_sum = 0
    while terminal is False:
        action = e_greedy_policy(q1 + q2, current_state, 0.1)
        next_observables, reward, terminal, info = env.step(action)
        env.render()
        rewards_sum += reward
        print(rewards_sum)
        time.sleep(0.5)
