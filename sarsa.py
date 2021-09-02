# -*- coding: utf-8 -*-

import gym
import numpy as np
from tqdm import tqdm
"""
Implementation of SARSA. Pseudocode in page 130 of the book "Reinforcement
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
        current_action = e_greedy_policy(q, current_state, epsilon)
        while terminal is False:
            next_state, reward, terminal, info = env.step(current_action)
            next_action = e_greedy_policy(q, current_state, epsilon)
            q[current_state][current_action] += alpha * (reward + gamma * q[next_state][next_action] - \
                                                q[current_state][current_action])
            current_state = next_state
            current_action = next_action
    return q

env = gym.make('NChain-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
q = run(n_states, n_actions, 1000, env, 0.99, 0.5, 0.9)
print(q)