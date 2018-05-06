#!/usr/bin/env python

"""
launch_manual.py
Launch the Grid Minecraft game with manual control.
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

import pdb


def main():
  """
  Program entry.
  """

  # Initialize the environment
  env = GridMinecraftEnv()
  env.reset()
  env.render()

  while True:
    # accept input
    key = input('MOVE (W,S,A,D) > ')
    if key.upper() == 'W':
      a = 0
    elif key.upper() == 'S':
      a = 1
    elif key.upper() == 'A':
      a = 2
    elif key.upper() == 'D':
      a = 3
    else:
      exit()

    # environment roll forward
    s_next, r, done, info = env.step(a)
    env.render()
    print('r = %f' % (r))
    print('')

    # check game over
    if done:
      print('Game Over.')
      exit()


if __name__ == '__main__':
  main()
