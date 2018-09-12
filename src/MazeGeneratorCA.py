#!/usr/bin/env python

"""
MazeGeneratorCA.py
Grid Minecraft automated maze generator with cellular automata.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."


import numpy as np
from scipy.signal import convolve2d as conv2d


class MazeGeneratorCA(object):
    """
    Grid Minecraft maze automated generator with cellular automata.
    """

    def __init__(self):
        """
        Initialize a maze generator.
        """

        super(MazeGeneratorCA, self).__init__()


    def _ca_step(self, X):
        """
        Evolve the cellular automata for one time step.

        @param X The current state of the cellular automata.
        @return The evolved cellular automata.
        """

        k = np.ones((3, 3)) # counter kernel

        N = conv2d(X, k, mode='same', boundary='wrap') - X # neighbour counts

        Y = (N == 3) | (X & (N > 0) & (N < 6)) # new state

        return Y


    def _ca_init(self, options={}):
        """
        Initialize cellular automata.

        @param options The options for initialization. (optional)
        """

        h = options['h'] # maze height
        w = options['w'] # maze width
        mh = options['mh'] # initialization template height
        mw = options['mw'] # initialization template width

        # initialize maze
        X_0 = np.zeros((h, w)).astype(bool)

        # generate random initialization template
        R = np.random.random((mh, mw)) > 0.75
        X_0[(h//2-mh//2):(h//2+mh//2), (w//2-mw//2):(w//2+mw//2)] = R

        return X_0


    def _label_cells_once(self, L):
        """
        Label cells once.

        @param L The previous cell labels.
        @return The updated cell labels by one update.
        """

        L = np.array(L)

        def bounded_min(L, spot, neighbour):
            if (0 <= neighbour[0] and neighbour[0] < L.shape[0] and
                0 <= neighbour[1] and neighbour[1] < L.shape[1]):
                if L[neighbour[0], neighbour[1]] > 0 and L[neighbour[0], neighbour[1]] < L[spot[0], spot[1]]:
                    return L[neighbour[0], neighbour[1]]
            return L[spot[0], spot[1]]

        for i in range(len(L)):
            for j in range(len(L[i])):
                L[i][j] = bounded_min(L, (i,j), (i-1,j))
                L[i][j] = bounded_min(L, (i,j), (i+1,j))
                L[i][j] = bounded_min(L, (i,j), (i,j-1))
                L[i][j] = bounded_min(L, (i,j), (i,j+1))

        return L


    def _label_cells(self, L):
        """
        Label all cells.

        @param L The initial cell labels.
        @return The updated cell labels.
        """

        L_prev = np.array(L)
        L = self._label_cells_once(L)

        while not np.all(L == L_prev):
            L_prev = np.array(L)
            L = self._label_cells_once(L)

        return L


    def _break_blocking_cell_once(self, L):
        """
        Break one blocking cell.

        @param L The current cell labels.
        @param The cell labels after breaking one blocking cell.
        """

        L = np.array(L)

        def break_cell(L, spot, neighbour1, neighbour2):
            if (0 <= neighbour1[0] and neighbour1[0] < L.shape[0] and
                0 <= neighbour1[1] and neighbour1[1] < L.shape[1] and
                0 <= neighbour2[0] and neighbour2[0] < L.shape[0] and
                0 <= neighbour2[1] and neighbour2[1] < L.shape[1] and
                L[neighbour1[0], neighbour1[1]] > 0 and
                L[neighbour2[0], neighbour2[1]] > 0 and
                L[neighbour1[0], neighbour1[1]] != L[neighbour2[0], neighbour2[1]]):
                return min(L[neighbour1[0], neighbour1[1]], L[neighbour2[0], neighbour2[1]])
            return L[spot[0], spot[1]]

        for i in range(len(L)):
            for j in range(len(L[i])):
                u = break_cell(L, (i,j), (i-1,j), (i+1,j))
                if u != L[i][j]:
                    L[i][j] = u
                    return L
                u = break_cell(L, (i,j), (i,j-1), (i,j+1))
                if u != L[i][j]:
                    L[i][j] = u
                    return L

        return L


    def _cvt_ca_to_labels(self, X):
        """
        Convert cellular automata states to labels.

        @param X The current state of the cellular automata.
        @return The labels for the cellular automata.
        """

        L = np.zeros(X.shape).astype(int)

        max_idx = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j]:
                    max_idx = max_idx + 1
                    L[i][j] = max_idx
                else:
                    L[i][j] = -1

        L = self._label_cells(L)

        return L


    def _cvt_labels_to_ca(self, L):
        """
        Convert labels to cellular automata states.

        @param X The current state of the cellular automata.
        @return The labels for the cellular automata.
        """

        X = np.zeros(L.shape).astype(bool)

        for i in range(len(X)):
            for j in range(len(X[i])):
                if L[i][j] > 0:
                    X[i][j] = True
                else:
                    X[i][j] = False

        return X


    def _connect(self, X):
        """
        Connect all cells.

        @param X The current state of the cellular automata.
        @return The fully connected cellular automata.
        """

        X = np.array(X)

        L = self._cvt_ca_to_labels(X)

        L_prev = np.array(L)
        L = self._break_blocking_cell_once(L)
        L = self._label_cells(L)
        while not np.all(L == L_prev):
            L_prev = np.array(L)
            L = self._break_blocking_cell_once(L)
            L = self._label_cells(L)

        X = self._cvt_labels_to_ca(L)

        return X


    def _sym(self, X):
        """
        Convert a boolean maze to a symbolic maze.

        @param X The boolean maze.
        """

        Y = np.zeros((X.shape))
        Y = Y.astype(str)

        Y[X] = '.'
        Y[~X] = 'X'

        return Y


    def _generateOnce(self, options={}, verbose=False):
        """
        Generate a random Grid Minecraft maze without ensuring it is valid.

        @param options The options for maze generation. (optional)
        @param verbose The switch to enable verbose output. (optional)
        """

        n_ca_steps_max = options['max_steps']
        n_ca_steps = 0

        # initialize the cellular automata
        X_0 = self._ca_init(options)
        if verbose:
            print(self._sym(X_0))
            print('')

        # initial evolution of the cellular automata
        X_prev = np.array(X_0)
        X = self._ca_step(X_0)
        n_ca_steps = n_ca_steps + 1
        if verbose:
            print(self._sym(X))
            print('')

        # evolve until stable
        while n_ca_steps < n_ca_steps_max and (not np.all(X == X_prev)):
            X_prev = np.array(X)
            X = self._ca_step(X)
            n_ca_steps = n_ca_steps + 1
            if verbose:
                print(self._sym(X))
                print('')

        # connect all cells
        X = self._connect(X)
        if verbose:
            print(self._sym(X))
            print('')

        return X


    def _verify(self, X):
        """
        Verify a maze.

        @param X The maze to verify.
        @return A boolean indicating if the maze if valid.
        """

        L = self._cvt_ca_to_labels(X)

        max_idx = np.amax(L)
        if max_idx != 1:
            return False

        count_cells = np.sum(L[L>0])
        if count_cells < 10:
            return False

        return True


    def generate(self, options={}, verbose=False):
        """
        Generate a random Grid Minecraft maze.

        @param options The options for maze generation. (optional)
        @param verbose The switch to enable verbose output. (optional)
        """

        _options = {
            'w': 10,
            'h': 10,
            'mw': 6,
            'mh': 6,
            'max_steps': 100
        }

        # update options
        for o in _options:
            if o in options:
                _options[o] = options[o]

        if 'max_steps' not in options:
            _options['max_steps'] = (np.sqrt(_options['w'] * _options['h']) * 10)

        # generate maze until valid 
        X = self._generateOnce(options=_options, verbose=False)
        while not self._verify(X):
            X = self._generateOnce(options=_options, verbose=False)

        return self._sym(X)


def _testMazeGenerator():
    mazegen = MazeGeneratorCA()

    maze = mazegen.generate(options={'w': 16, 'h': 8})
    print(maze)


def _test():
    _testMazeGenerator()


if __name__ == '__main__':
    _test()

