#!/usr/bin/env python

"""
GridMinecraftEnv.py
Grid Minecraft discrete environment.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np

from gym import Env, spaces
from gym.utils import seeding


class GridMinecraftEnv(Env):
    """
    Grid Minecraft discrete environment class.

    Symbol definitions:
        'A': the actor;
        '.': path, where the actor can move;
        'X': obstacle, which will block the way of the actor;
        'C': cliff, where the actor can fall into and die;
        'K': key, with which the actor can open the chest;
        'B': chest, inside which the magic sword is locked;
        'M': monster, which can only be killed by the magic sword.
        'G': gate, which the actor needs to get through.
    """

    def __init__(self):
        """
        Initialize a new Grid Minecraft environment.
        """

        # define Grid Minecraft environment
        self.world_map = [['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
                          ['X', '.', '.', '.', '.', '.', '.', '.', '.', 'X', 'C'],
                          ['X', 'X', 'X', '.', 'X', 'X', 'X', 'X', '.', 'K', 'C'],
                          ['X', '.', '.', '.', '.', '.', '.', 'X', 'X', 'C', 'C'],
                          ['C', 'C', '.', 'X', 'X', '.', 'X', 'X', 'X', 'X', 'X'],
                          ['C', 'B', '.', 'X', '.', '.', '.', '.', 'M', 'G', 'X'],
                          ['C', 'C', 'C', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']]
        self.reviving_spot = (1,1)

        self.max_horizon = 100

        self.R_running_cost = -0.01
        self.R_dead_cost    = -10.0
        self.R_gate_reward  = +100.0

        # define action and observation spaces
        self.nS = 8**(len(self.world_map) * len(self.world_map[0]) + 2)
        self.nA = 4

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # define internal state
        self.cur_steps = None
        self.actor_map = None
        self.actor_spot = None
        self.actor_status = {'key': None, 'sword': None}

        # set up external information cache
        self.a = None # last action
        self.s = None # current state (after last action was taken)
        self.r = None # last reward (after last action was taken)
        self.done = None # done
        
        # set up random seed
        self.seed()

        # reset environment
        self.reset()


    def seed(self, seed=None):
        """
        Set the random seed.
        """

        self.np_random, seed = seeding.np_random(seed)

        return [seed]


    def _generate_observation(actor_map, actor_spot, actor_status):
        """
        Generate observation. Note that this function does not help validate 
        the consistency and logic between the internal state and the world map.

        @param actor_map The current actor map of the Grid Minecraft world.
        @param actor_spot The actor spot.
        @param actor_status The status of the actor.
        @return s The observation.
        @return actor_map_next The next actor map.
        """

        actor_map_next = []
        s = []

        # construct new actor map
        for row in range(len(actor_map)):
            actor_map_next.append([])
            for col in range(len(actor_map[row])):
                actor_map_next[row].append('.' if actor_map[row][col] == 'A' else actor_map[row][col])
        actor_map_next[actor_spot[0]][actor_spot[1]] = 'A'

        # construct observation by added status bar
        s = actor_map_next.copy()
        s.append([' ', ' '])
        if actor_status['key']:
            s[-1][0] = 'K'
        if actor_status['sword']:
            s[-1][1] = 'S'

        return s, actor_map_next


    def _render_env(s):
        """
        Render the environment with a string.

        @param s The state.
        @return A rendered string of the environment.
        """

        env_str = ''
        for row in range(len(s)):
            env_str += ''.join(s[row]) + '\n'

        return env_str


    def _decode_action(a):
        """
        Convert action code to name.

        @param a The action.
        @return The name of the action.
        """

        if a == 0:
            return 'up'
        elif a == 1:
            return 'down'
        elif a == 2:
            return 'left'
        elif a == 3:
            return 'right'

        assert(False)


    def render(self):
        """
        Render the environment.
        """

        print(GridMinecraftEnv._render_env(self.s))
        print('')


    def reset(self):
        """
        Reset the environment.
        """

        # reset internal state
        self.cur_steps = 0
        self.actor_spot = tuple(self.reviving_spot)
        self.actor_status['key'] = False
        self.actor_status['sword'] = False

        # construct observation
        self.s, self.actor_map = GridMinecraftEnv._generate_observation(
            self.world_map, self.actor_spot, self.actor_status)

        # initialize other external information
        self.a = None
        self.r = None
        self.done = False

        return GridMinecraftEnv._render_env(self.s)


    def step(self, a):
        """
        Generate the next step of the environment.

        @param a The action to take.
        @return A tuple indicating the environment transition, defined as 
                ```
                (s, r, done, info)
                ```
                where `s` is the next observation of the environment, `r` is 
                the reward received by the agent, `done` indicates if the 
                environment is at an absorbing state, and `info` is the 
                internal information for debugging.
        """

        s = None
        r = None
        done = None

        self.cur_steps += 1

        # check if the environment is at an absorbing state
        if self.done:
            return (GridMinecraftEnv._render_env(self.s), 0, self.done, {
                'actor_spot': self.actor_spot, 
                'actor_status': self.actor_status})

        # construct the expected next action spot
        spot = None
        action_name = GridMinecraftEnv._decode_action(a)
        if action_name   == 'up':
            spot = (self.actor_spot[0] - 1, self.actor_spot[1]    )
        elif action_name == 'down':
            spot = (self.actor_spot[0] + 1, self.actor_spot[1]    )
        elif action_name == 'left':
            spot = (self.actor_spot[0]    , self.actor_spot[1] - 1)
        elif action_name == 'right':
            spot = (self.actor_spot[0]    , self.actor_spot[1] + 1)
        else:
            assert(False)

        # check if the actor stays within the world map
        if (spot[0] >= 0 and spot[0] < len(self.world_map) and
            spot[1] >= 0 and spot[1] < len(self.world_map[0])):
            # get symbol from actor map
            symbol = self.actor_map[spot[0]][spot[1]]

            # check new spot
            if symbol   == '.':
                self.actor_spot = spot
                r = self.R_running_cost
                done = False
            
            elif symbol == 'X':
                self.actor_spot = self.actor_spot
                r = self.R_running_cost
                done = False
            
            elif symbol == 'C':
                self.actor_spot = spot
                r = self.R_dead_cost
                done = True
            
            elif symbol == 'K':
                if (not self.actor_status['key']):
                    self.actor_spot = spot
                    self.actor_status['key'] = True
                else:
                    self.actor_spot = self.actor_spot
                r = self.R_running_cost
                done = False

            elif symbol == 'B':
                if self.actor_status['key'] and (not self.actor_status['sword']):
                    self.actor_spot = spot
                    self.actor_status['key'] = False
                    self.actor_status['sword'] = True
                else:
                    self.actor_spot = self.actor_spot
                r = self.R_running_cost
                done = False

            elif symbol == 'M':
                self.actor_spot = spot
                if self.actor_status['sword']:
                    r = self.R_running_cost
                    done = False
                else:
                    r = self.R_dead_cost
                    done = True
            
            elif symbol == 'G':
                self.actor_spot = spot
                r = self.R_gate_reward
                done = True
            
            else:
                assert(False)

        else:
            # spot not remains if actor gets out of the world map
            self.actor_spot = self.actor_spot
            r = self.R_running_cost
            done = False

        # generate new state and actor map
        s, self.actor_map = GridMinecraftEnv._generate_observation(
            self.actor_map, self.actor_spot, self.actor_status)

        # check horizon
        if self.cur_steps >= self.max_horizon:
            done = True

        # update cache
        self.a = a
        self.s = s
        self.r = r
        self.done = done

        return (GridMinecraftEnv._render_env(self.s), self.r, self.done, {
            'actor_spot': self.actor_spot,
            'actor_status': self.actor_status})

