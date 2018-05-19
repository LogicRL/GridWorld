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


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', dest='max_episodes',
                        type=int, default=1000,
                        help="Maximum number of episodes to run the LogicRL agent.")

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

    return parser.parse_args()


def main():
  args = parse_arguments()

  fname_domain = '../PDDL/domain.pddl'
  fname_problem = '../PDDL/problem_gridmc.pddl'
  
  # initialize environment
  env = GridMinecraftEnv()

  # initialize symbolic state encoder
  encoder = SymbolicStateEncoder()

  # initialize agent
  static_predicate_operators = [
    'freeGateExists',
    'guardedGateExists'
  ]
  agent = LogicRLAgent(
    env, encoder, fname_domain, fname_problem, static_predicate_operators,
    epsilon_max=1.0, epsilon_decay=1e-3, epsilon_min=0.01, lr=0.01, gamma=1.0)

  success = agent.autoplay(
    max_episodes=args.max_episodes, 
    pause_plan=args.plan, render=args.render, verbose=False)
  print('success: %s' % (success))

  # run validation episode
  agent.runEpisode(learn=False, render=True, verbose=True)


if __name__ == '__main__':
  main()

