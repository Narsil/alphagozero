from random import choice
from conf import conf
import numpy as np
import itertools
from math import cos, sin, pi

SIZE = conf['SIZE']

def _id(tensor):
    return tensor

def rotation_indexes(angle):
    rotation_swap = [0 for i in range(SIZE * SIZE + 1)]
    for x, y in itertools.product(range(SIZE), repeat=2):
        index = x + SIZE * y

        x = x - (SIZE - 1)/2
        y = y - (SIZE - 1)/2
        newx = cos(angle)*x - sin(angle)*y
        newy = sin(angle)*x + cos(angle)*y
        newx += (SIZE - 1)/2
        newy += (SIZE - 1)/2
        transpose_index = int(round(newx + SIZE * newy))
        rotation_swap[index] = transpose_index
    rotation_swap[SIZE * SIZE] = SIZE * SIZE
    return rotation_swap

def axis_symmetry_indexes(angle):
    rotation_swap = [0 for i in range(SIZE * SIZE + 1)]
    for x, y in itertools.product(range(SIZE), repeat=2):
        index = x + SIZE * y

        x = x - (SIZE - 1)/2
        y = y - (SIZE - 1)/2
        newx = cos(2*angle)*x + sin(2*angle)*y
        newy = sin(2*angle)*x - cos(2*angle)*y
        newx += (SIZE - 1)/2
        newy += (SIZE - 1)/2
        transpose_index = int(round(newx + SIZE * newy))
        if transpose_index > SIZE * SIZE:
            import ipdb;ipdb.set_trace()
        rotation_swap[index] = transpose_index
    rotation_swap[SIZE * SIZE] = SIZE * SIZE
    return rotation_swap

########## LEFT DIAGONAL
def left_diagonal(board):
    return np.transpose(board, axes=(0, 2, 1, 3))

LEFT_DIAGONAL_SWAP = axis_symmetry_indexes(pi/4.)
def reverse_left_diagonal(policy):
    policy[:,:] = policy[:,LEFT_DIAGONAL_SWAP]
    return policy

########## VERTICAL AXIS
def vertical_axis(board):
    board[:,:,list(range(SIZE)),:] = board[:,:,list(reversed(range(SIZE))),:]
    return board

VERTICAL_AXIS_SWAP = axis_symmetry_indexes(pi/2.)
def reverse_vertical_axis(policy):
    policy[:,:] = policy[:,VERTICAL_AXIS_SWAP]
    return policy

########## RIGHT DIAGONAL
def right_diagonal(board):
    return np.rot90(np.transpose(board, axes=(0, 2, 1, 3)), k=2, axes=(1, 2))

RIGHT_DIAGONAL_SWAP = axis_symmetry_indexes(3*pi/4.)
def reverse_right_diagonal(policy):
    policy[:,:] = policy[:,RIGHT_DIAGONAL_SWAP]
    return policy


########## HORIZONTAL AXIS
def horizontal_axis(board):
    board[:,list(range(SIZE)),:,:] = board[:,list(reversed(range(SIZE))),:,:]
    return board

HORIZONTAL_AXIS_SWAP = [0 for i in range(SIZE * SIZE + 1)]
for x, y in itertools.product(range(SIZE), repeat=2):
    index = x + SIZE * y
    transpose_index = x + SIZE * (SIZE - 1 - y)
    HORIZONTAL_AXIS_SWAP[index] = transpose_index
HORIZONTAL_AXIS_SWAP[SIZE * SIZE] = SIZE * SIZE
def reverse_horizontal_axis(policy):
    policy[:,:] = policy[:,HORIZONTAL_AXIS_SWAP]
    return policy

########### ROTATION 90
def rotation_90(board):
    return np.rot90(board, axes=(1, 2))

ROTATION_90_SWAP = rotation_indexes(pi/2.)
def reverse_rotation_90(policy):
    policy[:,:] = policy[:,ROTATION_90_SWAP]
    return policy

########### ROTATION 180
def rotation_180(board):
    return np.rot90(board, k=2, axes=(1, 2))

ROTATION_180_SWAP = rotation_indexes(pi)
def reverse_rotation_180(policy):
    policy[:,:] = policy[:,ROTATION_180_SWAP]
    return policy

########### ROTATION 270
def rotation_270(board):
    return np.rot90(board, k=3, axes=(1, 2))

ROTATION_270_SWAP = rotation_indexes(3*pi/2)
def reverse_rotation_270(policy):
    policy[:,:] = policy[:,ROTATION_270_SWAP]
    return policy


SYMMETRIES = [
    (_id, _id),
    (left_diagonal, reverse_left_diagonal),
    (vertical_axis, reverse_vertical_axis),
    (horizontal_axis, reverse_horizontal_axis),
    (rotation_90, reverse_rotation_90),
    (rotation_180, reverse_rotation_180),
    (rotation_270, reverse_rotation_270),
]

def random_symmetry_predict(model, board):
    symmetry, reverse_symmetry = choice(SYMMETRIES)
    symm_board = symmetry(board)
    symm_policy, value = model.predict_on_batch(symm_board)
    policy = reverse_symmetry(symm_policy)
    return policy, value
