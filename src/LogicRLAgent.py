#!/usr/bin/env python

"""
LogicRLAgent.py
LogicRL agent for automatic playing and learning.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import sys
import time
import argparse
import numpy as np
import gym
from collections import deque
from SymbolicStateEncoder import SymbolicStateEncoder
from PDDL import PDDLPlanner, show_plan
from RLAgents import RLAgents

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pdb, IPython


def print_symbolic_state_transition(s1, s2, prefix=''):
  """
  Print the difference between two symbolic states.

  @param s1 The original symbolic state.
  @param s2 The updated symbolic state.
  @param prefix The prefix added to the front of each line.
  """

  predicates_neg = s1.difference(s2)
  predicates_pos = s2.difference(s1)

  for p in predicates_neg:
    print(prefix + '- ' + str(p))

  for p in predicates_pos:
    print(prefix + '+ ' + str(p))


class LogicRLAgent(object):
  """
  LogicRLAgent class.
  """

  def __init__(self, env, encoder, fname_domain, fname_problem, static_predicate_operators, epsilon_max=1.0, epsilon_decay=1e-3, epsilon_min=0.01, lr=0.01, gamma=1.0):
    super(LogicRLAgent, self).__init__()
    
    self.env = env
    self.fname_domain = fname_domain
    self.fname_problem = fname_problem

    self.epsilon_max = epsilon_max
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min

    self.lr = lr
    self.gamma = gamma

    self.agent_running_cost   = -1
    self.agent_error_state_cost = -3
    self.agent_subgoal_reward = 100
    self.agent_failure_cost   = -100

    # initialize a symbolic planner
    self.planner = PDDLPlanner(fname_domain, fname_problem)
    self.static_predicate_operators = static_predicate_operators

    # construct predefined initial state and static predicates
    self.static_predicates = set()
    self.predefined_initial_state = set()
    for a in self.planner.predefined_initial_state:
      # construct initial state
      self.predefined_initial_state.add(tuple(a.predicate))

      # construct static predicates
      if a.predicate[0] in self.static_predicate_operators:
        self.static_predicates.add(tuple(a.predicate))

    # construct predefined goals
    self.predefined_goals = set()
    for a in self.planner.predefined_goals:
      self.predefined_goals.add(tuple(a.predicate))

    # initialize a symbolic state encoder
    self.encoder = encoder

    # initialize a RLAgents pool
    self.agents = RLAgents(self.env.action_space.n)
    

  def encodeSymbolicState(self, s):
    """
    Encode symbolic state from a lower level state.

    @param s The lower-level state to encode.
    @return The corresponding symbolic state as a set of predicates.
    """

    ss = self.encoder.encode(s)

    ss = ss.union(self.static_predicates)

    return ss


  def predictActionByAgent(self, agent_name, s):
    """
    Predict an action to take by an agent at a specific state.

    @param agent_name The name of the agent.
    @param s The lower-level state.
    @return The lower-level action predicted by the agent to take.
    """

    learn_count = self.agents.statistics(agent_name)['learn_count']
    epsilon = max(self.epsilon_max - self.epsilon_decay * learn_count, self.epsilon_min)

    a = self.agents.act(agent_name, s, epsilon=epsilon)

    return a


  def feedbackToAgent(self, agent_name, s, a, s_next, r, done):
    """
    Provide feedback to an RL agent.

    @param agent_name The name of the RL agent.
    @param s The RL state the agent was at.
    @param a The lower-level action taken by the agent.
    @param s_next The RL state the agent reached after taking the action.
    @param r The reward received by the agent.
    @param lr The learning rate.
    @param gamma The reward decay rate.
    @param done A boolean indicating if an episode ends.
    """

    self.agents.learn(agent_name, s, a, s_next, r, lr=self.lr, gamma=self.gamma)


  def findSymbolicPlan(self, ss):
    """
    Find a symbolic plan towards the goal.

    @param ss The symbolic state to start from.
    @return A plan found. `None` will be returned if no plan is found.
    """

    plan = self.planner.find_plan(initial_state=ss, goals=None) # using default goals

    return plan


  def runEpisode(self, learn=True, render=False, render_sleep=0.5, verbose=False):
    """
    Run an episode.

    @param learn The switch to enable online learning.
    @param render The switch to enable rendering.
    @param render_sleep The interval to sleep after rendering.
    @param verbose The switch to enable verbose log.
    @return success A boolean indicating if the agent solve the game within the maximum 
                    number of episodes.
    @return lst_r The reward sequence given to the lower-level reinforcement 
                  learning agent.
    @return lst_r_env The environment reward sequence.
    """

    env = self.env
    g = self.predefined_goals # already converted to predicate sets

    lst_r = []
    lst_r_env = []

    # reset the environment
    s = env.reset()
    ss = self.encodeSymbolicState(s)

    # render if requested
    if render:
      env.render()
      time.sleep(render_sleep)

    # find initial symbolic plan
    plan = self.findSymbolicPlan(ss)

    # loop run
    done = False
    while not done:
      # check if a feasible plan exists
      if plan is None or len(plan) == 0:
        done = True
        if verbose:
          print('[ INFO ] failed to find feasible plan')
        continue

      # check if goals already satisfied
      if len(plan) == 1:
        assert(len(plan[0][1].intersection(g)) == len(g))
        done = True
        if verbose:
          print('[ INFO ] subgoal satisfied')
        return True, lst_r, lst_r_env
        
      # extract states and operator from plan
      ss_cur = plan[0][1]
      op_next = plan[1][0]
      ss_next_expected = plan[1][1]

      # predict the lower-level action to take
      agent_name = op_next
      a = self.predictActionByAgent(agent_name, s)

      # execute the lower-level action
      s_next, r_env, done, info = env.step(a)
      ss_next = self.encodeSymbolicState(s_next)

      # print state transition
      if verbose:
        if len(ss.difference(ss_next)) > 0 or len(ss_next.difference(ss)) > 0:
          print_symbolic_state_transition(ss, ss_next)

      # determine reward for RL agent
      r = 0
      if ss_next == ss_cur and not done:
        # assign subtask reward
        r = self.agent_running_cost

        # print verbose message
        if verbose:
          print('[ INFO ] symbolic state remains (r: %f, op: %s)' % (r, op_next))

      elif ss_next == ss_next_expected:
        # assign subtask reward
        r = self.agent_subgoal_reward

        # print verbose message
        if verbose:
          print('[ INFO ] symbolic plan step executed (r: %f, op: %s)' % (r, op_next))

        # replan due to symbolic state change
        plan = self.findSymbolicPlan(ss_next)

      else:
        # assign subtask reward
        r = self.agent_failure_cost
        done = True

        # print verbose message
        if verbose:
          print('[ INFO ] subtask failed (r: %f, op: %s)' % (r, op_next))

      # render if requested
      if render:
        env.render()
        time.sleep(render_sleep)

      # feedback to agent
      if learn:
        self.feedbackToAgent(agent_name, s, a, s_next, r, done)

      # record rewards
      lst_r.append(r)
      lst_r_env.append(r_env)

      # check if goals satisfied
      if len(plan) == 1:
        if len(ss_next.intersection(g)) == len(g):
          done = True
          if verbose:
            print('[ INFO ] subgoal satisfied')
          return True, lst_r, lst_r_env

      # update states
      s = s_next
      ss = ss_next

    return False, lst_r, lst_r_env


  def autoplay(self, max_episodes, pause_plan=False, render=False, verbose=False):
    """
    Play autonomously and learn online.

    @param max_episodes The maximum number of episodes to run the LogicRLAgent.
    @param pause_plan The switch to pause while showing the initial plan.
    @param render The switch to enable rendering.
    @param verbose The switch to enable verbose log.
    @return success A boolean indicating if the agent solve the game within the maximum 
                    number of episodes.
    @return lst_R The return sequence for lower-level agents.
    @return lst_R_env The return sequence from environment.
    @rturn lst_steps The sequence of numbers of steps until done.
    """

    # print the initial symbolic plan
    s = self.env.reset()
    ss = self.encodeSymbolicState(s)
    plan = self.findSymbolicPlan(ss)
    print('initial plan:')
    show_plan(plan)
    if pause_plan:
      print('')
      input('press ENTER to start autoplay..')
    print('')

    # loop through the episodes
    success = False
    lst_R = []
    lst_R_env = []
    lst_steps = []
    for episode in range(max_episodes):
      if verbose:
        print('')
        print('[ INFO ] episode: %d / %d' % (episode, max_episodes))

      episode_success, lst_r, lst_r_env = self.runEpisode(learn=True, render=render, verbose=verbose)
      R = np.sum(lst_r)
      R_env = np.sum(lst_r_env)
      
      lst_R.append(R)
      lst_R_env.append(R_env)
      lst_steps.append(len(lst_r_env))

      if not verbose:
        print('[ INFO ] episode: %d / %d, R: %f, R_env: %f' % (
          episode, max_episodes, R, R_env))

      if episode_success:
        success = True

    return success, lst_R, lst_R_env, lst_steps


def main():
  pass


if __name__ == '__main__':
  main()
