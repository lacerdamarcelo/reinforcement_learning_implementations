# -*- coding: utf-8 -*-
import gym
import numpy as np
"""
Implementation of the algorithm off-policy Monte Carlo control for estimation
of the optimal policy. Pseudocode in page 111 of the book "Reinforcement
Learning: an Instroduction", 2nd edition, by Richard S. Sutton and Andrew G.
Barto.
"""
def run(n_states, n_actions, max_episodes, env, gamma, epsilon):
    q = np.zeros((n_states, n_actions))
    c = np.zeros((n_states, n_actions))
    pi_policy = np.zeros(n_states)
    for state in range(n_states):
        pi_policy[state] = np.argmax(q[state])
    for episode in range(max_episodes):
        b_policy = np.full((n_states, n_actions), epsilon / n_actions)
        for state in range(n_states):
            b_policy[state][np.argmax(q[state])] =  1 - epsilon + epsilon / n_actions
        terminal = False
        state = env.reset()
        states = [state]
        rewards = []
        actions = []
        while terminal is False:
            action = np.random.choice(list(range(n_actions)), p=b_policy[state])
            state, reward, terminal, info = env.step(action)
            rewards.append(reward)
            actions.append(action)
            if terminal is False:
                states.append(state)
        G = 0
        W = 1
        for n_step in np.arange(len(states) - 1, -1, -1):
            G = gamma * G + rewards[n_step]
            c[states[n_step]][actions[n_step]] += W
            q[states[n_step]][actions[n_step]] += (W / c[states[n_step]][actions[n_step]]) * (G - q[states[n_step]][actions[n_step]])
            pi_policy[states[n_step]] = np.argmax(q[states[n_step]])
            # WHY IS THAT??
            if actions[n_step] != pi_policy[states[n_step]]:
                break
            W = W * (1 / b_policy[states[n_step]][actions[n_step]])
    print(q)
    return pi_policy

env = gym.make('NChain-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
print(run(n_states, n_actions, 100, env, 0.99, 0.3))