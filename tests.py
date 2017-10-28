# -*- coding: utf-8 -*-
import unittest
import numpy as np
from self_play import (
        color_board, _get_points, capture_group, make_play, legal_moves,
)

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

    def test_taking_stones(self):
        board = np.array(
                         [[0, 1, 0],
                          [1,-1, 1],
                          [0, 1, 0]]) 
        target_group = [(1, 1)]
        group = capture_group(1, 1, board)
        self.assertEquals(group, target_group)

    def test_taking_group_stones(self):
        board = np.array(
                         [[0, 1, 0],
                          [1,-1, 1],
                          [1,-1, 1],
                          [0, 1, 0]]) 
        target_group = [(1, 1), (1, 2)]
        group = capture_group(1, 1, board)
        self.assertEquals(group, target_group)

        target_group = [(1, 2), (1, 1)]
        group = capture_group(1, 2, board)
        self.assertEquals(group, target_group)

    def test_taking_group_stones_sides(self):
        board = np.array(
                         [[-1, 1, 0],
                          [ 1, 0, 0],
                          [ 0, 0, 0]])
        target_group = [(0, 0)]
        group = capture_group(0, 0, board)
        self.assertEquals(group, target_group)

        board = np.array(
                         [[-1,-1, 1],
                          [ 1, 1, 0],
                          [ 0, 0, 0]])
        target_group = [(0, 0), (1, 0)]
        group = capture_group(0, 0, board)
        self.assertEquals(group, target_group)

        target_group = [(1, 0), (0, 0)]
        group = capture_group(1, 0, board)
        self.assertEquals(group, target_group)

    def test_circle_group(self):
        board = np.array(
                         [[ 0, 1, 1, 1, 0],
                          [ 1,-1,-1,-1, 1],
                          [ 1,-1, 1,-1, 1],
                          [ 1,-1,-1,-1, 1],
                          [ 0, 1, 1, 1, 0]])
        target_group = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
        self.assertEquals(len(target_group), 8)
        for x, y in target_group:
            group = capture_group(x, y, board)
            self.assertEquals(sorted(group), sorted(target_group))

class TestBoardMethods(unittest.TestCase):
    def test_legal_moves_ko(self):
        board = np.zeros((1, 19, 19, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player

        make_play(0, 0, board) # black
        make_play(1, 0, board) # white
        make_play(1, 1, board) # black
        make_play(2, 1, board) # white
        make_play(8, 8, board) # black random pos
        make_play(3, 0, board) # white
        # ○ ● . ● . .
        # . ○ ● . . .
        # . . . . . .
        make_play(2, 0, board) # black captures_first
        # ○ . ○ ● . .
        # . ○ ● . . .
        # . . . . . .
        mask = legal_moves(board)
        self.assertEquals(board[0][0][1][0], 0) # white stone
        self.assertEquals(board[0][0][1][1], 0) # was taken
        self.assertEquals(board[0][0][1][2], 1) # white stone was here
        self.assertEquals(board[0][0][1][3], 0) # black stone was not here
        self.assertEquals(mask[1], True)

    def test_legal_moves_not_ko(self):
        board = np.zeros((1, 19, 19, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player

        make_play(0, 0, board) # black
        make_play(1, 0, board) # white
        make_play(1, 1, board) # black
        make_play(2, 0, board) # white
        make_play(2, 1, board) # black
        make_play(8, 8, board) # white random pos
        # ○ ● ● . . .
        # . ○ ○ . . .
        # . . . . . .
        make_play(3, 0, board) # black captures_first
        # ○ . . ○ . .
        # . ○ ○ . . .
        # . . . . . .
        mask = legal_moves(board)
        self.assertEquals(board[0][0][1][0], 0) # white stone 1
        self.assertEquals(board[0][0][1][1], 0) # was taken
        self.assertEquals(board[0][0][2][0], 0) # white stone 2
        self.assertEquals(board[0][0][2][1], 0) # was taken
        self.assertEquals(board[0][0][1][2], 1) # white stone 1 was here
        self.assertEquals(board[0][0][1][3], 0) # black stone was not here
        self.assertEquals(board[0][0][2][2], 1) # white stone 2 was here
        self.assertEquals(board[0][0][2][3], 0) # black stone was not here
        self.assertEquals(mask[1], False)
        self.assertEquals(mask[2], False)



if __name__ == '__main__':
    unittest.main()
