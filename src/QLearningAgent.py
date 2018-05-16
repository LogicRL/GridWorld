#!/usr/bin/env python

"""
QLearningAgent.py
Q-learning agent for discrete environment with dynamic state and transition 
learning. The environment shall be stationary in terms of state transition and 
reward mechanism.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np
import gym
import time

import pdb, IPython


class QLearningAgent(object):
  """
  Q-learning agent class.
  """

  def __init__(self, n_action):
    """
    Initialize a Q-learning agent.

    @param n_action The number of possible actions to take.
    """

    super(QLearningAgent, self).__init__()

    self.n_action = n_action

    self.Q = dict() # state-action value model - {s: [q, ...]}


  def act(self, s, epsilon=0):
    """
    Predict the next action at a state.

    @param s The state the agent is at.
    @param epsilon The exploration randomness.
    """

    a_random = int(np.floor(np.random.random() * 4))

    if s not in self.Q:
      self.Q[s] = [0] * self.n_action

    a = np.argmax(self.Q[s])

    if np.random.random() < epsilon:
      a = a_random

    return a


  def learn(self, s, a, s_next, r, lr=0.01, gamma=1.0):
    """
    Learn new experience.

    @param s The original state.
    @param a The action taken.
    @param s_next The state transitted to.
    @param r The reward received.
    @param lr The learning rate.
    @param gamma The reward decay rate.
    """

    if s not in self.Q:
      self.Q[s] = [0] * self.n_action

    if s_next not in self.Q:
      self.Q[s_next] = [0] * self.n_action

    Q_update = r + gamma * np.max(self.Q[s_next])

    self.Q[s][a] += lr * (Q_update - self.Q[s][a])

