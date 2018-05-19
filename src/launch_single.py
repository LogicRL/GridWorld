#!/usr/bin/env python

"""
launch_single.py
Launch the Grid Minecraft game with single agent.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np
import gym
import time
from GridMinecraftEnv import GridMinecraftEnv
from QLearningAgent import QLearningAgent
from MBAgent import MBAgent

import pdb, IPython


def print_state_action_values(Q):
  """
  Print state-action values.
  """

  for s in Q:
    print(s)
    print(Q[s])
    print('')


def run_episode(env, agent, epsilon=0, lr=0.01, gamma=1.0, learn=True, render=False, verbose=False):
  """
  Run the agent for an episode in the environment.

  @param env The environment.
  @param agent The agent to run.
  @param epsilon The exploration randomness.
  @param lr The learning rate.
  @param gamma The reward decay rate.
  @param learn The switch to turn on learning.
  @param render The switch to turn on rendering.
  @param verbose The switch to turn on verbose logging.
  @return acc_rewards The accumulative environment rewards.
  """

  acc_rewards = 0

  s = env.reset()
  if render:
    env.render()
    time.sleep(0.5)

  done = False
  while not done:
    # choose an action
    a = agent.act(s, epsilon=epsilon)

    # environment roll forward
    s_next, r, done, info = env.step(a)
    acc_rewards += r

    # learn experience
    if learn:
      agent.learn(s, a, s_next, r, lr=lr, gamma=gamma)
    
    # update state
    s = s_next

    if verbose:
      print('r = %f' % (r, acc_rewards))
      print('')

    if render:
      env.render()
      time.sleep(0.1)

  return acc_rewards


def main():
  """
  Program entry.
  """

  # initialize the environment
  env = GridMinecraftEnv()

  # initialize agent
  agent = QLearningAgent(env.action_space.n)
  #agent = MBAgent(env.action_space.n)

  # run episodes
  episodes = 1000
  epsilon_max = 0.2
  epsilon_decay = 1e-3
  epsilon_min = 0.01
  lst_return = []
  for episode in range(episodes):
    epsilon = max(epsilon_max - epsilon_decay * episode, epsilon_min)
    
    render = False
    if episode == episodes - 1:
      render = True

    acc_rewards = run_episode(env, agent, 
      epsilon=epsilon, lr=0.01, gamma=0.99, 
      learn=True, render=render, verbose=False)
    lst_return.append(acc_rewards)

    print('episode: %d/%d, acc_rewards: %f, epsilon: %f' % (episode+1, episodes, acc_rewards, epsilon))

    if episode == episodes - 1:
      IPython.embed()
    
  print('average return: %f' % (np.mean(lst_return)))


if __name__ == '__main__':
  main()
