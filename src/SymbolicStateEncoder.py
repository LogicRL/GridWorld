#!/usr/bin/env python

"""
SymbolicStateEncoder.py
The encoder that encodes lower-level states to symbolic states for the Grid 
Minecraft environment.
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

class SymbolicStateEncoder(object):
  """
  Symbolic state encoder class.
  """

  def __init__(self):
    """
    Initialize a symbolic state encoder.
    """

    super(SymbolicStateEncoder, self).__init__()
    
  
  def encode(self, s):
    """
    Encode a lower-level state to a symbolic state.

    @param s The lower-level states.
    @return The symbolic state as a set of detected predicates.
    """

    s = s.split('\n')

    ss = set()

    # detect variable predicates
    if s[2][9] == 'K':
      ss.add(tuple(['keyExists', 'key1']))

    if s[5][1] == 'B':
      ss.add(tuple(['chestExists', 'chest1']))

    if s[5][8] == 'M':
      ss.add(tuple(['monsterExists', 'monster1']))

    if s[7][0] == 'K':
      ss.add(tuple(['actorWithKey']))

    if s[7][1] == 'S':
      ss.add(tuple(['actorWithSword']))

    if s[5][9] == 'A':
      ss.add(tuple(['actorReachedGate']))

    return ss

