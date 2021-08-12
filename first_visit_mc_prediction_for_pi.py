# -*- coding: utf-8 -*-
import gym
import numpy as np
"""
Implementation of the algorithm first-visit Monte Carlo control for estimation
of the optimal policy. Pseudocode in page 101 of the book "Reinforcement
Learning: an Instroduction", 2nd edition, by Richard S. Sutton and Andrew G.
Barto. A few adaptations have been made in order to make it more efficient.
"""
def run(policy, initial_v, n_states, n_actions, max_episodes, env, gamma,
        epsilon):
    sum_rewards_per_state_action = np.zeros((n_states, n_actions))
    rewards_counters_per_state_action = np.full((n_states, n_actions),
                                                0.000000001)
    for episode in range(max_episodes):
        terminal = False
        state = env.reset()
        states = [state]
        rewards = []
        actions = []
        while terminal is False:
            action = np.random.choice(list(range(n_actions)), p=policy[state])
            state, reward, terminal, info = env.step(action)
            rewards.append(reward)
            actions.append(action)
            if terminal is False:
                states.append(state)
        G = 0        
        current_reward_per_state_action = np.full((n_states, n_actions),
                                                  None)
        for n_step in np.arange(len(states) - 1, -1, -1):
            G = gamma * G + rewards[n_step]
            current_reward_per_state_action[states[n_step]][actions[n_step]] = G
        for state, rewards in enumerate(current_reward_per_state_action):
            current_q_values = np.zeros(n_actions)
            for action, reward in enumerate(current_reward_per_state_action[state]):
                if reward is not None:
                    sum_rewards_per_state_action[state][action] += reward
                    rewards_counters_per_state_action[state][action] += 1
            current_q_values = sum_rewards_per_state_action[state] / \
                               rewards_counters_per_state_action[state]
            optimal_action = np.argmax(current_q_values)
            policy[state] = epsilon / n_actions
            policy[state][optimal_action] = 1 - epsilon + epsilon / n_actions
    return policy

env = gym.make('NChain-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
# Using a random policy
policy = np.full((n_states, n_actions), 1 / n_actions)
print(run(policy, 0, n_states, n_actions, 10000, env, 0.99, 0.2))