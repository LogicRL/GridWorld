#!/usr/bin/env python

"""
launch_logicRL.py
Launch the Grid Minecraft game with LogicRL agent.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np
import gym
import time
import argparse
from GridMinecraftEnv import GridMinecraftEnv
from SymbolicStateEncoder import SymbolicStateEncoder
from PDDL import PDDLPlanner
from LogicRLAgent import LogicRLAgent

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pdb, IPython


def plt_errorbar_compressed(data_len, compress_size, Y, E):
  X_comp = []
  Y_comp = []
  E_comp = []

  for i in range(int(data_len / compress_size)):
    X_comp.append(i*compress_size)
    Y_comp.append(np.mean(Y[i*compress_size:(i+1)*compress_size]))
    E_comp.append(np.mean(E[i*compress_size:(i+1)*compress_size]))

  plt.errorbar(X_comp, Y_comp, E_comp, capsize=3)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', dest='max_episodes',
                        type=int, default=1000,
                        help="Maximum number of episodes to run the LogicRL agent.")
    parser.add_argument('--trials', dest='trials',
                        type=int, default=10,
                        help="Number of trials to run in experiment mode.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--plan', dest='plan',
                              action='store_true',
                              help="Whether to pause while showing the initial plan.")
    parser_group.add_argument('--skip-plan', dest='plan',
                              action='store_false',
                              help="Whether to pause while showing the initial plan.")
    parser.set_defaults(plan=False)

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--exp', dest='exp',
                              action='store_true',
                              help="Whether to turn on experiment mode.")
    parser_group.add_argument('--demo', dest='exp',
                              action='store_false',
                              help="Whether to turn on experiment mode.")
    parser.set_defaults(plan=False)

    return parser.parse_args()


def main():
  args = parse_arguments()

  fname_domain = '../PDDL/domain.pddl'
  fname_problem = '../PDDL/problem_gridmc.pddl'
  
  # initialize environment
  env = GridMinecraftEnv()

  # initialize symbolic state encoder
  encoder = SymbolicStateEncoder()

  # define static names
  static_predicate_names = [
    'freeGateExists',
    'guardedGateExists'
  ]

  # demo mode
  if not args.exp:
    agent = LogicRLAgent(
      env, encoder, fname_domain, fname_problem, static_predicate_names,
      epsilon_max=1.0, epsilon_decay=1e-3, epsilon_min=0.01, lr=0.01, gamma=1.0)

    success, lst_R, lst_R_env, lst_steps = agent.autoplay(
      max_episodes=args.max_episodes, 
      pause_plan=args.plan, render=args.render, verbose=False)
    print('success: %s' % (success))

    # run validation episode
    agent.runEpisode(learn=False, render=True, verbose=True)

  # experiment mode
  if args.exp:
    mat_R = []
    mat_R_env = []
    mat_steps = []

    # run trials
    for trial in range(args.trials):
      agent = LogicRLAgent(
        env, encoder, fname_domain, fname_problem, static_predicate_names,
        epsilon_max=1.0, epsilon_decay=1e-3, epsilon_min=0.01, lr=0.01, gamma=1.0)

      success, lst_R, lst_R_env, lst_steps = agent.autoplay(
        max_episodes=args.max_episodes, 
        pause_plan=args.plan, render=args.render, verbose=False)

      mat_R.append(lst_R)
      mat_R_env.append(lst_R_env)
      mat_steps.append(lst_steps)

    mat_R = np.array(mat_R)
    mat_R_env = np.array(mat_R_env)
    mat_steps = np.array(mat_steps)

    # draw graphs
    mean_R = np.mean(mat_R, 0)
    std_R = np.std(mat_R, 0)
    mean_R_env = np.mean(mat_R_env, 0)
    std_R_env = np.std(mat_R_env, 0)
    mean_steps = np.mean(mat_steps, 0)
    std_steps = np.std(mat_steps, 0)

    compress_size = 20

    plt.figure(1)
    plt_errorbar_compressed(len(mean_R), compress_size, mean_R, std_R)
    plt.xlabel('episode')
    plt.ylabel('accumulated rewards of lower-level agents')
    
    plt.figure(2)
    plt_errorbar_compressed(len(mean_R_env), compress_size, mean_R_env, std_R_env)
    plt.xlabel('episode')
    plt.ylabel('accumulated environmental rewards')

    plt.figure(3)
    plt_errorbar_compressed(len(mean_steps), compress_size, mean_steps, std_steps)
    plt.xlabel('episode')
    plt.ylabel('steps until done')

    plt.show()

  IPython.embed()


if __name__ == '__main__':
  main()

