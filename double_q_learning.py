# -*- coding: utf-8 -*-

import os
import gym
import time
import numpy as np
from tqdm import tqdm
import plotly.express as px
"""
Implementation of Double Q-Learning. Pseudocode in page 136 of the book "Reinforcement
Learning: an Instroduction", 2nd edition, by Richard S. Sutton and Andrew G.
Barto.
"""

def e_greedy_policy(q, current_state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(q[current_state]))
    else:
        return np.argmax(q[current_state])

def run(n_states, n_actions, max_episodes, env, gamma, epsilon, alpha):
    q1 = np.zeros((n_states, n_actions))
    q2 = np.zeros((n_states, n_actions))
    rewards_sums = []
    for episode in tqdm(range(max_episodes)):
        terminal = False
        current_state = env.reset()
        reward_sum = 0
        while terminal is False:
            action = e_greedy_policy(q1 + q2, current_state, epsilon)
            next_state, reward, terminal, info = env.step(action)
            if np.random.random() > 0.5:
                q1[current_state][action] += alpha * (reward + gamma * q2[next_state][np.argmax(q1[next_state])] - \
                                             q1[current_state][action])
            else:
                q2[current_state][action] += alpha * (reward + gamma * q1[next_state][np.argmax(q2[next_state])] - \
                                             q2[current_state][action])
            current_state = next_state
            reward_sum += reward
        rewards_sums.append(reward_sum)
    return q1, q2, rewards_sums

env = gym.make('FrozenLake-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
q1, q2, rewards_sums = run(n_states, n_actions, 100000, env, 0.99, 0.1, 0.9)
q = q1 + q2
fig = px.line(x=range(len(rewards_sums)), y=rewards_sums)
fig.show()

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