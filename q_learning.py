# -*- coding: utf-8 -*-

import os
import gym
import time
import numpy as np
from tqdm import tqdm
"""
Implementation of Q-Learning. Pseudocode in page 131 of the book "Reinforcement
Learning: an Instroduction", 2nd edition, by Richard S. Sutton and Andrew G.
Barto.
"""

def e_greedy_policy(q, current_state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(q[current_state]))
    else:
        return np.argmax(q[current_state])

def run(n_states, n_actions, max_episodes, env, gamma, epsilon, alpha):
    q = np.zeros((n_states, n_actions))
    for episode in tqdm(range(max_episodes)):
        terminal = False
        current_state = env.reset()
        while terminal is False:
            action = e_greedy_policy(q, current_state, epsilon)
            next_state, reward, terminal, info = env.step(action)
            q[current_state][action] += alpha * (reward + gamma * np.max(q[next_state]) - \
                                        q[current_state][action])
            current_state = next_state
    return q

env = gym.make('Taxi-v3')
n_states = env.observation_space.n
n_actions = env.action_space.n
q = run(n_states, n_actions, 10000, env, 0.99, 0.1, 0.9)

for i in range(10):
    state = env.reset()
    terminal = False
    while terminal is False:
        env.render()
        state, reward, terminal, info = env.step(np.argmax(q[state]))
        time.sleep(0.1)
        os.system('cls' if os.name == 'nt' else 'clear')
    env.render()
    time.sleep(0.1)
    os.system('cls' if os.name == 'nt' else 'clear')