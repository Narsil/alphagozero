# -*- coding: utf-8 -*-
from conf import conf
conf['SIZE'] = 9  # Override settings for tests
conf['KOMI'] = 5.5  # Override settings for tests

import unittest
import numpy as np
from self_play import (
        color_board, _get_points, capture_group, make_play, legal_moves,
        index2coord, simulate, play_game,
)
from symmetry import (
        _id,
        left_diagonal, reverse_left_diagonal,
        right_diagonal, reverse_right_diagonal,
        vertical_axis, reverse_vertical_axis,
        horizontal_axis, reverse_horizontal_axis,
        rotation_90, reverse_rotation_90,
        rotation_180, reverse_rotation_180,
        rotation_270, reverse_rotation_270,
)
import itertools

class TestGoMethods(unittest.TestCase):
    def assertEqualList(self, arr1, arr2):
        self.assertEqual(arr1.tolist(), arr2.tolist())

    def test_coloring_player_1(self):
        board = np.array(
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]) 
        target = np.array(
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) 
        self.assertEqualList(color_board(board, 1), target)
        board = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 0, 1, -1,  0, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        target = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 1, 1, -1,  0, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        self.assertEqualList(color_board(board, 1), target)

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
        self.assertEqualList(total, target)

    def test_coloring_player_2(self):
        board = np.array(
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]) 
        target = np.array(
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]) 
        self.assertEqualList(color_board(board, -1), target)
        board = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 0, 1, -1,  0, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        target = np.array(
                         [[1, 1, 1, -1, -1, -1],
                          [1, 0, 1, -1, -1, -1],
                          [1, 1, 1, -1, -1, -1]]) 
        self.assertEqualList(color_board(board, -1), target)

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
        self.assertEqual(_get_points(board), {0: 29, 1: 12, 2: 11, -1: 15, -2: 14})

    def test_taking_stones(self):
        board = np.array(
                         [[0, 1, 0],
                          [1,-1, 1],
                          [0, 1, 0]]) 
        target_group = [(1, 1)]
        group = capture_group(1, 1, board)
        self.assertEqual(group, target_group)

    def test_taking_group_stones(self):
        board = np.array(
                         [[0, 1, 0],
                          [1,-1, 1],
                          [1,-1, 1],
                          [0, 1, 0]]) 
        target_group = [(1, 1), (1, 2)]
        group = capture_group(1, 1, board)
        self.assertEqual(group, target_group)

        target_group = [(1, 2), (1, 1)]
        group = capture_group(1, 2, board)
        self.assertEqual(group, target_group)

    def test_taking_group_stones_sides(self):
        board = np.array(
                         [[-1, 1, 0],
                          [ 1, 0, 0],
                          [ 0, 0, 0]])
        target_group = [(0, 0)]
        group = capture_group(0, 0, board)
        self.assertEqual(group, target_group)

        board = np.array(
                         [[-1,-1, 1],
                          [ 1, 1, 0],
                          [ 0, 0, 0]])
        target_group = [(0, 0), (1, 0)]
        group = capture_group(0, 0, board)
        self.assertEqual(group, target_group)

        target_group = [(1, 0), (0, 0)]
        group = capture_group(1, 0, board)
        self.assertEqual(group, target_group)

    def test_taking_group_sucide(self):
        board = np.array(
                         [[-1, 1, 0],
                          [ 1, 0, 0],
                          [ 0, 0, 0]])
        target_group = [(0, 0)]
        group = capture_group(0, 0, board)
        self.assertEqual(group, target_group)

        board = np.array(
                         [[-1,-1, 1],
                          [ 1, 1, 0],
                          [ 0, 0, 0]])
        target_group = [(0, 0), (1, 0)]
        group = capture_group(0, 0, board)
        self.assertEqual(group, target_group)

        target_group = [(1, 0), (0, 0)]
        group = capture_group(1, 0, board)
        self.assertEqual(group, target_group)

    def test_circle_group(self):
        board = np.array(
                         [[ 0, 1, 1, 1, 0],
                          [ 1,-1,-1,-1, 1],
                          [ 1,-1, 1,-1, 1],
                          [ 1,-1,-1,-1, 1],
                          [ 0, 1, 1, 1, 0]])
        target_group = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
        self.assertEqual(len(target_group), 8)
        for x, y in target_group:
            group = capture_group(x, y, board)
            self.assertEqual(sorted(group), sorted(target_group))

class TestBoardMethods(unittest.TestCase):
    def test_self_sucide(self):
        board = np.zeros((1, 19, 19, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player

        make_play(0, 0, board) # black
        make_play(1, 0, board) # white
        make_play(8, 9, board) # black random
        make_play(2, 1, board) # white
        make_play(8, 8, board) # black random pos
        make_play(3, 0, board) # white
        # ○ ● . ● . .
        # . . ● . . .
        # . . . . . .
        make_play(2, 0, board) # black sucides
        self.assertEqual(board[0][0][1][0], 1) # white stone
        self.assertEqual(board[0][0][1][1], 0) # was not taken

        self.assertEqual(board[0][0][2][0], 0) # black stone
        self.assertEqual(board[0][0][2][1], 0) # was taken

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
        self.assertEqual(board[0][0][1][0], 0) # white stone
        self.assertEqual(board[0][0][1][1], 0) # was taken
        self.assertEqual(board[0][0][1][2], 1) # white stone was here
        self.assertEqual(board[0][0][1][3], 0) # black stone was not here
        self.assertEqual(mask[1], True)

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
        self.assertEqual(board[0][0][1][0], 0) # white stone 1
        self.assertEqual(board[0][0][1][1], 0) # was taken
        self.assertEqual(board[0][0][2][0], 0) # white stone 2
        self.assertEqual(board[0][0][2][1], 0) # was taken
        self.assertEqual(board[0][0][1][2], 1) # white stone 1 was here
        self.assertEqual(board[0][0][1][3], 0) # black stone was not here
        self.assertEqual(board[0][0][2][2], 1) # white stone 2 was here
        self.assertEqual(board[0][0][2][3], 0) # black stone was not here
        self.assertEqual(mask[1], False)
        self.assertEqual(mask[2], False)

    def test_full_board_capture(self):
        size = conf['SIZE']
        board = np.zeros((1, size, size, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player

        for i in range(size*size - 2):
            x, y = index2coord(i)
            make_play(x, y, board) # black
            make_play(0, size, board) # white pass

        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ . .
        make_play(0, size, board) # black pass
        make_play(size -1, size - 1, board) # white corner
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ . ●

        for i in range(size*size - 2):
            x, y = index2coord(i)
            self.assertEqual(board[0][y][x][0], 1) # black stone i
            self.assertEqual(board[0][y][x][1], 0) # black stone i

        self.assertEqual(board[0][size - 1][size - 1][0], 0) # white stone
        self.assertEqual(board[0][size - 1][size - 1][1], 1) # white stone
        self.assertEqual(board[0][size - 1][size - 2][0], 0) # empty
        self.assertEqual(board[0][size - 1][size - 2][1], 0) # empty

        make_play(size - 2, size - 1, board) # black
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ ○
        # ○ ○ ○ ○ ○ .
        for i in range(size*size - 1):
            x, y = index2coord(i)
            self.assertEqual(board[0][y][x][0], 0) # black stone i
            self.assertEqual(board[0][y][x][1], 1) # black stone i (it's white's turn)
        self.assertEqual(board[0][size - 1][size - 1][0], 0) # empty
        self.assertEqual(board[0][size - 1][size - 1][1], 0) # empty

        make_play(size - 1, size - 1, board) # white
        # . . . . . .
        # . . . . . .
        # . . . . . .
        # . . . . . ●
        for i in range(size*size - 1):
            x, y = index2coord(i)
            self.assertEqual(board[0][y][x][0], 0) # empty
            self.assertEqual(board[0][y][x][1], 0) # empty
        self.assertEqual(board[0][size - 1][size - 1][0], 0) # white
        self.assertEqual(board[0][size - 1][size - 1][1], 1) # white

    def test_bug(self):
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ○ ○ ● ●
        # ● ● ● ● ● ● . ● ●
        # ● ● ● ● ● ● ○ ○ ○
        size = conf['SIZE']
        board = np.zeros((1, size, size, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player

        for i in range(size*size):
            x, y = index2coord(i)
            if (x, y) in [(5, 6), (6, 6), (6, 8), (7, 8), (8, 8)]:
                make_play(x, y, board) # black
                make_play(0, size, board) # white pass
            elif (x, y) in [(6, 7)]:
                make_play(0, size, board) # black pass
                make_play(0, size, board) # white pass
            else:
                make_play(0, size, board) # black pass
                make_play(x, y, board) # white

        make_play(0, size, board) # black pass
        make_play(6, 7, board) # white

        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● . . ● ●
        # ● ● ● ● ● ● ● ● ●
        # ● ● ● ● ● ● . . . 
        for i in range(size*size - 1):
            x, y = index2coord(i)
            if (x, y) in [(5, 6), (6, 6), (6, 8), (7, 8), (8, 8)]:
                self.assertEqual(board[0][y][x][0], 0) # empty
                self.assertEqual(board[0][y][x][1], 0) # emtpy
            else:
                self.assertEqual(board[0][y][x][0], 0) # white
                self.assertEqual(board[0][y][x][1], 1) # white

class TestSymmetrydTestCase(unittest.TestCase):

    def setUp(self):
        size = conf['SIZE']
        board = np.zeros((1, size, size, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player
        policy = np.zeros((1, size * size + 1), dtype=np.float32)
        self.board = board
        self.size = size
        self.policy = policy
        board = self.board


        for x, y in [(1, 1), (1, 2), (1, 3), (2, 3)]:
            make_play(x, y, board) # black
            make_play(0, size, board) # white pass
            policy[0, x + y * size] = 1

        policy[0, size * size] = -1  # Pass move

    def test_id(self):
        board = self.board
        size = self.size

        old_board = np.copy(board)
        board = _id(board)
        
        for i, j in zip(old_board.reshape(-1), board.reshape(-1)):
            self.assertEqual(i, j)

        policy = np.arange(size*size + 1)
        old_policy = np.copy(policy)
        policy = _id(policy)

        for i, j in zip(old_policy.reshape(-1), policy.reshape(-1)):
            self.assertEqual(i, j)

    def test_left_diagonal(self):
        board = self.board
        size = self.size
        should_be_ones = [(1, 1), (2, 1), (3, 1), (3, 2)] # Transposed

        board = left_diagonal(board)

        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_left_diagonal(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)

    def test_vertical_axis(self):
        board = self.board
        size = self.size
        should_be_ones = [(7, 1), (7, 2), (7, 3), (6, 3)] # vertical_axis

        board = vertical_axis(board)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_vertical_axis(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)

    def test_right_diagonal(self):
        board = self.board
        size = self.size
        should_be_ones = [(7, 7), (6, 7), (5, 7), (5, 6)] #  right diagonal

        board = right_diagonal(board)

        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_right_diagonal(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)


    def test_horizontal_axis(self):
        board = self.board
        size = self.size
        should_be_ones = [(1, 7), (1, 6), (1, 5), (2, 5)] # horizontal_axis

        board = horizontal_axis(board)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_horizontal_axis(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)

    def test_rotation_90(self):
        board = self.board
        size = self.size
        should_be_ones = [(1, 7), (2, 7), (3, 7), (3, 6)] # Rotation 90deg anticlockwise

        board = rotation_90(board)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_rotation_90(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)

    def test_rotation_180(self):
        board = self.board
        size = self.size
        should_be_ones = [(7, 7), (7, 6), (7, 5), (6, 5)] # Rotation 180deg anticlockwise

        board = rotation_180(board)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_rotation_180(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)

    def test_rotation_270(self):
        board = self.board
        size = self.size
        should_be_ones = [(7, 1), (6, 1), (5, 1), (5, 2)] # Rotation 270deg anticlockwise

        board = rotation_270(board)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(board[0,y,x,0], 1)
                self.assertEqual(board[0,y,x,1], 0)
            else:
                self.assertEqual(board[0,y,x,0], 0)
                self.assertEqual(board[0,y,x,1], 0)

        policy = self.policy
        policy = reverse_rotation_270(policy)
        for x, y in itertools.product(range(size), repeat=2):
            if (x, y) in should_be_ones:
                self.assertEqual(policy[0, x + size * y], 1)
            else:
                self.assertEqual(policy[0, x + size * y], 0)
        self.assertEqual(policy[0, size * size], -1)


class DummyModel(object):
    name = "dummy_model"
    def predict(self, X):
        policies, values = self.predict_on_batch(X)
        return policies[0], values[0]

    def predict_on_batch(self, X):
        size = conf['SIZE']
        batch_size = X.shape[0]
        policy = np.zeros((batch_size, size * size + 1), dtype=np.float32)
        for i in range(batch_size):
            policy[i,:] = 1./(size*size +1)

        value = np.zeros((batch_size, 1), dtype=np.float32)
        value[:] = 1
        return policy, value

class MCTSTestCase(unittest.TestCase):
    def setUp(self):
        # Remove the symmetries for reproductibility
        import symmetry
        symmetry.SYMMETRIES = symmetry.SYMMETRIES[0:1]
        size = conf['SIZE']
        tree = {
            'count': 0,
            'mean_value': 0,
            'value': 0,
            'parent': None,
            'subtree': {
                0:{
                    'count': 0,
                    'p': 1,
                    'value': 0,
                    'mean_value': 0,
                    'subtree': {}
                }, 
                1: {
                    'count': 0,
                    'p': 0,
                    'mean_value': 0,
                    'value': 0,
                    'subtree': {}
                }
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][1]['parent'] = tree
        
        board = np.zeros((1, size, size, 17), dtype=np.float32)
        player = 1
        board[:,:,:,-1] = player

        model = DummyModel()

        self.model = model
        self.board = board
        self.tree = tree

    def test_leaf(self):
        tree = self.tree
        board = self.board 
        model = self.model

        simulate(tree, board, model, mcts_batch_size=2)
        self.assertEqual(tree['subtree'][0]['count'], 1)
        self.assertEqual(tree['subtree'][1]['count'], 1)
        self.assertEqual(tree['subtree'][0]['value'], 1)
        self.assertEqual(tree['subtree'][1]['value'], 1)
        self.assertEqual(tree['count'], 2)
        self.assertEqual(tree['value'], 2)


    def test_model_evaluation(self):
        tree = self.tree
        board = self.board 
        size = conf['SIZE']

        test_board1 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board1[:,:,:,-1] = 1
        make_play(0, 0, test_board1)

        test_board2 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board2[:,:,:,-1] = 1
        make_play(1, 0, test_board2)

        class DummyModel(object):
            def predict_on_batch(_, X):
                size = conf['SIZE']
                board1 = X[0].reshape(1, size, size, 17)
                board2 = X[1].reshape(1, size, size, 17)
                self.assertTrue(np.array_equal(board1, test_board1))
                self.assertTrue(np.array_equal(board2, test_board2))
                batch_size = X.shape[0]
                policy = np.zeros((batch_size, size * size + 1), dtype=np.float32)
                policy[:,0] = 1

                value = np.zeros((batch_size, 1), dtype=np.float32)
                value[:] = 1
                return policy, value
        model = DummyModel()

        simulate(tree, board, model, mcts_batch_size=2)

    def test_model_evaluation_nested(self):
        tree = {
            'count': 0,
            'mean_value': 0,
            'value': 0,
            'parent': None,
            'subtree':{
                0:{
                    'count': 0,
                    'p': 1,
                    'value': 0,
                    'mean_value': 0,
                    'subtree': {
                        1: {       # <----- This will be checked first
                            'count': 0,
                            'p': 1,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        },
                        2: {       # <----- This will be checked second
                            'count': 0,
                            'p': 0,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        }
                    }
                }, 
                1: {
                    'count': 0,
                    'p': 0,
                    'mean_value': 0,
                    'value': 0,
                    'subtree': {},
                }
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][0]['subtree'][1]['parent'] = tree['subtree'][0]
        tree['subtree'][0]['subtree'][2]['parent'] = tree['subtree'][0]
        tree['subtree'][1]['parent'] = tree

        board = self.board 
        size = conf['SIZE']

        test_board1 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board1[:,:,:,-1] = 1
        make_play(0, 0, test_board1)
        make_play(1, 0, test_board1)

        test_board2 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board2[:,:,:,-1] = 1
        make_play(0, 0, test_board2)
        make_play(2, 0, test_board2)

        class DummyModel(object):
            def predict_on_batch(_, X):
                size = conf['SIZE']
                board1 = X[0].reshape(1, size, size, 17)
                board2 = X[1].reshape(1, size, size, 17)
                self.assertTrue(np.array_equal(board1, test_board1))
                self.assertTrue(np.array_equal(board2, test_board2))
                batch_size = X.shape[0]
                policy = np.zeros((batch_size, size * size + 1), dtype=np.float32)
                policy[:,0] = 1

                value = np.zeros((batch_size, 1), dtype=np.float32)
                value[:] = 1
                return policy, value
        model = DummyModel()
        # Remove the symmetries for reproductibility

        simulate(tree, board, model, mcts_batch_size=2)

    def test_model_evaluation_other_nested(self):
        tree = {
            'count': 0,
            'mean_value': 0,
            'value': 0,
            'parent': None,
            'subtree':{
                0:{
                    'count': 0,
                    'p': 1,
                    'value': 0,
                    'mean_value': 0,
                    'subtree': {},
                }, 
                1: {
                    'count': 0,
                    'p': 0,
                    'mean_value': 0,
                    'value': 0,
                    'subtree': {
                        0: {
                            'count': 0,
                            'p': 0,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        },
                        2: {
                            'count': 0,
                            'p': 1,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        }
                    }
                }
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][1]['parent'] = tree
        tree['subtree'][1]['subtree'][0]['parent'] = tree['subtree'][1]
        tree['subtree'][1]['subtree'][2]['parent'] = tree['subtree'][1]

        board = self.board 
        size = conf['SIZE']

        test_board1 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board1[:,:,:,-1] = 1
        make_play(0, 0, test_board1)

        test_board2 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board2[:,:,:,-1] = 1
        make_play(1, 0, test_board2)
        make_play(2, 0, test_board2)

        class DummyModel(object):
            def predict_on_batch(_, X):
                size = conf['SIZE']
                board1 = X[0].reshape(1, size, size, 17)
                board2 = X[1].reshape(1, size, size, 17)
                self.assertTrue(np.array_equal(board1, test_board1))
                self.assertTrue(np.array_equal(board2, test_board2))
                batch_size = X.shape[0]
                policy = np.zeros((batch_size, size * size + 1), dtype=np.float32)
                policy[:,0] = 1

                value = np.zeros((batch_size, 1), dtype=np.float32)
                value[:] = 1
                return policy, value
        model = DummyModel()

        simulate(tree, board, model, mcts_batch_size=2)

    def test_small_batch_size(self):
        tree = self.tree
        model = self.model
        board = self.board 

        simulate(tree, board, model, mcts_batch_size=1)
        self.assertEqual(tree['subtree'][0]['count'], 1)
        self.assertEqual(tree['subtree'][0]['value'], 1)
        self.assertNotEqual(tree['subtree'][0]['subtree'], {})

        self.assertEqual(tree['subtree'][1]['count'], 0)
        self.assertEqual(tree['subtree'][1]['value'], 0)
        self.assertEqual(tree['subtree'][1]['subtree'], {})

    def test_nested_selected(self):
        model = self.model
        board = self.board 

        tree = {
            'count': 0,
            'mean_value': 0,
            'value': 0,
            'parent': None,
            'subtree':{
                0:{
                    'count': 0,
                    'p': 1,
                    'value': 0,
                    'mean_value': 0,
                    'subtree': {
                        1: {
                            'count': 0,
                            'p': 0,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        },
                        2: {
                            'count': 0,
                            'p': 1,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        }
                    }
                }, 
                1: {
                    'count': 0,
                    'p': 0,
                    'mean_value': 0,
                    'value': 0,
                    'subtree': {},
                }
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][0]['subtree'][1]['parent'] = tree['subtree'][0]
        tree['subtree'][0]['subtree'][2]['parent'] = tree['subtree'][0]
        tree['subtree'][1]['parent'] = tree

        simulate(tree, board, model, mcts_batch_size=2)
        self.assertEqual(tree['subtree'][0]['count'], 2)
        self.assertEqual(tree['subtree'][0]['subtree'][1]['count'], 1)
        self.assertEqual(tree['subtree'][0]['subtree'][2]['count'], 1)
        self.assertEqual(tree['subtree'][1]['count'], 0)
        self.assertEqual(tree['subtree'][0]['value'], 2)
        self.assertEqual(tree['subtree'][0]['mean_value'], 1)
        self.assertEqual(tree['subtree'][1]['value'], 0)

    def test_nested_other_leaves(self):
        model = self.model
        board = self.board 

        tree = {
            'count': 0,
            'mean_value': 0,
            'value': 0,
            'parent': None,
            'subtree': {
                0:{
                    'count': 0,
                    'p': .75,
                    'value': 0,
                    'mean_value': 0,
                    'subtree': {}
                }, 
                1: {
                    'count': 0,
                    'p': .25,
                    'mean_value': 0,
                    'value': 0,
                    'subtree': {
                        0: {
                            'count': 0,
                            'p': 1,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        },
                        2: {
                            'count': 0,
                            'p': 0,
                            'mean_value': 0,
                            'value': 0,
                            'subtree': {},
                        }
                    }
                },
                2:{
                    'count': 0,
                    'p': 0,
                    'value': 0,
                    'mean_value': 0,
                    'subtree': {}
                }, 
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][1]['parent'] = tree
        tree['subtree'][1]['subtree'][0]['parent'] = tree['subtree'][1]
        tree['subtree'][1]['subtree'][2]['parent'] = tree['subtree'][1]

        simulate(tree, board, model, mcts_batch_size=2)
        self.assertEqual(tree['subtree'][0]['count'], 1)
        self.assertEqual(tree['subtree'][0]['value'], 1)
        self.assertEqual(tree['subtree'][1]['value'], 1)
        self.assertEqual(tree['subtree'][1]['count'], 1)
        self.assertEqual(tree['subtree'][1]['subtree'][0]['count'], 1)
        self.assertEqual(tree['subtree'][1]['subtree'][2]['count'], 0)
        self.assertEqual(tree['count'], 2)
        self.assertEqual(tree['mean_value'], 1)
        self.assertEqual(tree['subtree'][2]['count'], 0)
        self.assertEqual(tree['subtree'][2]['subtree'], {})

class PlayTestCase(unittest.TestCase):
    def setUp(self):
        # Remove the symmetries for reproductibility
        import symmetry
        symmetry.SYMMETRIES = symmetry.SYMMETRIES[0:1]

    def test_play(self):
        model = DummyModel()
        mcts_simulations = 8 # mcts batch size is 8 and we need at least one batch
        boards_and_policies, winner, _ = play_game(model, model, mcts_simulations, conf['STOP_EXPLORATION'], self_play=True, num_moves=2)

        size = conf['SIZE']
        test_board1 = np.zeros((1, size, size, 17), dtype=np.float32)
        test_board1[:,:,:,-1] = 1

        board, policy = boards_and_policies[0]
        self.assertTrue(np.array_equal(board, test_board1)) # First board is empty

        self.assertEqual(winner, 0)  # White should win with 5.5 komi after 2 moves

        for move, (board, policy_target) in enumerate(boards_and_policies[::2]): # Black player lost
            player = board[0,0,0,-1]
            value_target = 1 if winner == player else -1

            self.assertEqual(player, 1)
            self.assertEqual(value_target, -1)

        for move, (board, policy_target) in enumerate(boards_and_policies[1::2]): # White player won
            player = board[0,0,0,-1]
            value_target = 1 if winner == player else -1

            self.assertEqual(player, 0)
            self.assertEqual(value_target, 1)




if __name__ == '__main__':
    unittest.main()
