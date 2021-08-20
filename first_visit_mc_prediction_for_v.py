# -*- coding: utf-8 -*-
import gym
import numpy as np
from tqdm import tqdm
"""
Implementation of the algorithm first-visit Monte Carlo for estimation of V.
Pseudocode in page 92 of the book "Reinforcement Learning: an Instroduction",
2nd edition, by Richard S. Sutton and Andrew G. Barto. A few adaptations have
been made in order to make it more efficient.
"""
def estimate_v(policy, initial_v, n_states, max_episodes, env, gamma):
    sum_rewards_per_state = np.zeros(n_states)
    rewards_counters_per_state = np.full(n_states, 0.000000000001)
    for episode in tqdm(range(max_episodes)):
        terminal = False
        state = env.reset()
        states = [state]
        rewards = []
        while terminal is False:
            action = policy[state]
            state, reward, terminal, info = env.step(action)
            rewards.append(reward)
            if terminal is False:
                states.append(state)
        G = 0        
        current_reward_per_state = [None] * n_states
        for n_step in np.arange(len(states) - 1, -1, -1):
            G = gamma * G + rewards[n_step]
            current_reward_per_state[states[n_step]] = G
        for state, reward in enumerate(current_reward_per_state):
            if reward is not None:
                sum_rewards_per_state[state] += reward
                rewards_counters_per_state[state] += 1  
    return sum_rewards_per_state / rewards_counters_per_state

env = gym.make('NChain-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
# Using a random policy
policy = np.random.randint(low=0, high=n_actions, size=n_states)
print(estimate_v(policy, 0, n_states, 10000, env, 0.99))