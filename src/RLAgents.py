#!/usr/bin/env python

"""
RLAgents.py
RL agents for discrete environment with dynamic state and transition learning. 
The environment shall be stationary in terms of state transition and reward 
mechanism.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np
import gym
import time
from QLearningAgent import QLearningAgent

import pdb, IPython


class RLAgents(object):
  """
  Q-learning agent class.
  """

  def __init__(self, n_action, auto_add=True):
    """
    Initialize a Q-learning agent.

    @param n_action The number of possible actions to take.
    @param auto_add The switch to turn automatic agent adding.
    """

    super(RLAgents, self).__init__()

    self.n_action = n_action
    self.auto_add = auto_add

    self.agents = dict()
    self.agents_statistics = dict() # {'act_count': int, 'learn_count': int}


  def add_agent(self, agent_name):
    """
    Add a new agent.

    @param agent_name The name of the new agent. Note that if an agent already 
                      exists, this operation will be ignored.
    """

    if agent_name not in self.agents:
      self.agents[agent_name] = QLearningAgent(self.n_action)
      self.agents_statistics[agent_name] = {
        'act_count': 0,
        'learn_count': 0
      }


  def statistics(self, agent_name):
    """
    Retrive the statistics of an agent.

    @param agent_name The name of the agent.
    @return The statistics of the agent.
    """

    if agent_name not in self.agents and self.auto_add:
      self.add_agent(agent_name)

    return self.agents_statistics[agent_name]


  def act(self, agent_name, s, epsilon=0):
    """
    Predict the next action at a state by an agent.

    @param agent_name The name of the agent.
    @param s The state the agent is at.
    @param epsilon The exploration randomness.
    """

    if agent_name not in self.agents and self.auto_add:
      self.add_agent(agent_name)

    a = self.agents[agent_name].act(s, epsilon)

    self.agents_statistics[agent_name]['act_count'] += 1

    return a


  def learn(self, agent_name, s, a, s_next, r, lr=0.01, gamma=1.0):
    """
    Learn new experience for an agent.

    @param agent_name The name of the agent.
    @param s The original state.
    @param a The action taken.
    @param s_next The state transitted to.
    @param r The reward received.
    @param lr The learning rate.
    @param gamma The reward decay rate.
    """

    if agent_name not in self.agents and self.auto_add:
      self.add_agent(agent_name)

    self.agents[agent_name].learn(s, a, s_next, r, lr=lr, gamma=gamma)

    self.agents_statistics[agent_name]['learn_count'] += 1

