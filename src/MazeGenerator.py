#!/usr/bin/env python

"""
MazeGenerator.py
Grid Minecraft automated maze generator.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np

from MazeGeneratorCA import MazeGeneratorCA


class MazeGenerator(object):
  """
  Grid Minecraft automated maze generator.
  """

  def __init__(self):
    """
    Initialize a maze generator.
    """

    super(MazeGenerator, self).__init__()

    # select maze base generator
    self._mbgen = MazeGeneratorCA()


  def _find_free_spots(self, maze):
    """
    Find free spots in the maze.

    @param maze The current maze.
    @return A list of free spot indices.
    """

    spots = []
    for i in range(len(maze)):
      for j in range(len(maze[i])):
        if maze[i][j] == '.':
          spots.append((i,j))

    return spots


  def _find_free_corners(self, maze, strict=True):
    """
    Find free corner spots in the maze.

    @param maze The current maze.
    @param strict A boolean indicating if the obstacle must be wall.
    @return A list of free corner spot indices.
    """

    def is_wall(maze, spot, strict=True):
      if (0 <= spot[0] and spot[0] < maze.shape[0] and
          0 <= spot[1] and spot[1] < maze.shape[1]):
        if strict and maze[spot[0], spot[1]] != 'X':
          return False
        if (not strict) and maze[spot[0], spot[1]] == '.':
          return False
      return True

    def is_corner(maze, spot):
      if maze[spot[0], spot[1]] != '.':
        return False
      count_NS_walls = 0
      count_EW_walls = 0
      wall_N = False
      wall_S = False
      wall_E = False
      wall_W = False
      if is_wall(maze, (spot[0]-1, spot[1]), strict=strict):
        count_NS_walls += 1
        wall_N = True
      if is_wall(maze, (spot[0]+1, spot[1]), strict=strict):
        count_NS_walls += 1
        wall_S = True
      if is_wall(maze, (spot[0], spot[1]-1), strict=strict):
        count_EW_walls += 1
        wall_W = True
      if is_wall(maze, (spot[0], spot[1]+1), strict=strict):
        count_EW_walls += 1
        wall_E = True
      if count_EW_walls == 1 and count_NS_walls == 1:
        if wall_N and wall_E and maze[spot[0]+1, spot[1]-1] == '.':
          return True
        if wall_E and wall_S and maze[spot[0]-1, spot[1]-1] == '.':
          return True
        if wall_S and wall_W and maze[spot[0]-1, spot[1]+1] == '.':
          return True
        if wall_W and wall_N and maze[spot[0]+1, spot[1]+1] == '.':
          return True
      if ((count_EW_walls + count_NS_walls == 3) and 
          count_EW_walls > 0 and count_NS_walls > 0):
        return True
      return False

    corners = []
    for i in range(len(maze)):
      for j in range(len(maze[i])):
        if is_corner(maze, (i,j)):
          corners.append((i,j))

    return corners


  def _generate_gate(self, maze, prob_M=0.5):
    """
    Generate a gate in the maze.

    @param maze The current maze.
    @param prob_M The probability of generating a monster guarded gate.
    @return The updated maze.
    """

    maze = np.array(maze)

    corners = self._find_free_corners(maze, strict=True)

    spot = corners[np.random.choice(range(len(corners)))]

    maze[spot[0], spot[1]] = 'M' if np.random.random() < prob_M else 'G'

    return maze


  def _generate_chest(self, maze):
    """
    Generate a chest in the maze.

    @param maze The current maze.
    @return The updated maze.
    """

    maze = np.array(maze)

    corners = self._find_free_corners(maze, strict=False)

    spot = corners[np.random.choice(range(len(corners)))]

    maze[spot[0], spot[1]] = 'B'

    return maze


  def _generate_key(self, maze):
    """
    Generate a key in the maze.

    @param maze The current maze.
    @return The updated maze.
    """

    maze = np.array(maze)

    spots = self._find_free_spots(maze)

    spot = spots[np.random.choice(range(len(spots)))]

    maze[spot[0], spot[1]] = 'K'

    return maze


  def generate(self, options={}):
    """
    Generate a maze with filled in elements.

    @param options The maze generation options. (optional)
    @return A generated maze.
    """

    maze = self._mbgen.generate(options)

    # generate gate
    maze = self._generate_gate(maze)

    # generate chest
    maze = self._generate_chest(maze)

    # generate key
    maze = self._generate_key(maze)

    return maze


def _test_MazeGenerator():
  mazegen = MazeGenerator()

  maze = mazegen.generate(options={'w': 16, 'h': 8})

  print(maze)


def _test():
  _test_MazeGenerator()


if __name__ == '__main__':
  _test()
