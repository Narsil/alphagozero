import unittest
import numpy as np
from self_play import color_board, _get_points

class TestGoMethods(unittest.TestCase):
    def assertEqualsList(self, arr1, arr2):
        self.assertEquals(arr1.tolist(), arr2.tolist())

    def test_coloring_player_1(self):
        board = np.array(
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]) 
        target = np.array(
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) 
        self.assertEqualsList(color_board(board, 1), target)
        board = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 0, 1, -1,  0, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        target = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 1, 1, -1,  0, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        self.assertEqualsList(color_board(board, 1), target)

    def test_player_1_big(self):
        board = np.array([
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,], 
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,],
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,],
                 [0,  0,  0,  1, -1,  0,  0, -1,  0,],
                 [1,  1,  1, -1,  0, -1, -1,  0,  0,],
                 [0,  0,  0,  1, -1,  0,  0, -1, -1,],
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,],
                 [0,  0,  0,  1,  0, -1,  0,  1,  0,],
                 [0,  0,  0,  0,  0, -1,  0,  0,  0,],
        ])
        target = np.array([
                 [1,  1,  1,  2,  0, -2, -1, -1, -1,], 
                 [1,  1,  1,  2,  0, -2, -1, -1, -1,],
                 [1,  1,  1,  2,  0, -2, -1, -1, -1,],
                 [1,  1,  1,  2, -2, -1, -1, -2, -1,],
                 [2,  2,  2, -2, -1, -2, -2, -1, -1,],
                 [0,  0,  0,  2, -2,  0,  0, -2, -2,],
                 [0,  0,  0,  2,  0, -2,  0,  0,  0,],
                 [0,  0,  0,  2,  0, -2,  0,  2,  0,],
                 [0,  0,  0,  0,  0, -2,  0,  0,  0,],
        ]) 
        colored1 = color_board(board,  1)
        colored2 = color_board(board, -1)
        total = colored1 + colored2
        self.assertEqualsList(total, target)

    def test_coloring_player_2(self):
        board = np.array(
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]) 
        target = np.array(
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]) 
        self.assertEqualsList(color_board(board, -1), target)
        board = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 0, 1, -1,  0, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        target = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 0, 1, -1, -1, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        self.assertEqualsList(color_board(board, -1), target)

    def test_get_winner(self):
        board = np.array([
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,], 
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,],
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,],
                 [0,  0,  0,  1, -1,  0,  0, -1,  0,],
                 [1,  1,  1, -1,  0, -1, -1,  0,  0,],
                 [0,  0,  0,  1, -1,  0,  0, -1, -1,],
                 [0,  0,  0,  1,  0, -1,  0,  0,  0,],
                 [0,  0,  0,  1,  0, -1,  0,  1,  0,],
                 [0,  0,  0,  0,  0, -1,  0,  0,  0,],
        ])
        self.assertEquals(_get_points(board), {0: 29, 1: 12, 2: 11, -1: 15, -2: 14})

if __name__ == '__main__':
    unittest.main()
