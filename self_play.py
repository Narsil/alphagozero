# -*- coding: utf-8 -*-
import os
from conf import conf
from keras.models import load_model
from model import loss
import numpy as np
import numpy.ma as ma
from numpy.ma.core import MaskedConstant
import datetime
from math import sqrt

SWAP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
Cpuct = 1

def index2coord(index):
    y = index / 19
    x = index - 19 * y
    return x, y

def legal_moves(board):
    mask1 = board[0,:,:,0].reshape(-1) != 0
    mask2 = board[0,:,:,1].reshape(-1) != 0
    # TODO Deal with ko later
    mask = mask1 + mask2
    mask = np.append(mask, 0)
    return mask

def new_leaf(policy):
    leaf = {}
    for move, p in enumerate(policy.reshape(-1)):
        if isinstance(p, MaskedConstant):
            continue
        leaf[move] = {
            'count': 0,
            'value': 0,
            'mean_value': 0,
            'p': p,
            'subtree':{}
        }
    return leaf

def simulate(node, board, model):
    total_n = sqrt(sum(dic['count'] for dic in node.values()))
    if total_n == 0:
        total_n = 1
    # Select exploration
    max_a = -1
    max_v = -1
    for a, dic in node.items():
        u = Cpuct * dic['p'] * total_n / (1. + dic['count']) 
        v = dic['mean_value'] + u
        if v > max_v:
            max_v = v
            max_a = a

    selected_action = max_a
    selected_node = node[selected_action]
    x, y = index2coord(selected_action)
    player = board[0,0,0,-1]
    board, player = make_play(x, y, board)
    if selected_node['subtree'] != {}:
        value = simulate(selected_node['subtree'], board, model)
    else:
        # This is a leaf
        N = len(node)
        policy, value = model.predict(board)
        mask = legal_moves(board)
        policy = ma.masked_array(policy, mask=mask)
        leaf = new_leaf(policy)
        selected_node['subtree'] = leaf

    selected_node['count'] += 1
    selected_node['value'] += value
    selected_node['mean_value'] = selected_node['value'] / float(selected_node['count'])
    return value

def mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model):
    for i in range(mcts_simulations):
        test_board = np.copy(board)
        simulate(mcts_tree, test_board, model)

    if temperature == 1:
        total_n = sum(dic['count'] for dic in mcts_tree.values())
        moves = []
        ps = []
        for move, dic in mcts_tree.items():
            n = dic['count']
            if not n:
                continue
            p = dic['count'] / float(total_n)
            moves.append(move)
            ps.append(p)
        selected_a = np.random.choice(moves, size=1, p=ps)[0]
    elif temperature == 0:
        _, selected_a = max((dic['count'], a) for a, dic in mcts_tree.items())
    mask = legal_moves(board)
    return selected_a

def select_play(policy, board, mcts_simulations, mcts_tree, temperature, model):
    mask = legal_moves(board)
    policy = ma.masked_array(policy, mask=mask)
    index = mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model)

    # index = np.argmax(policy)
    x, y = index2coord(index)
    return index

def get_real_board(board):
    player = board[0,0,0,-1]
    if player == 1:
        real_board = board[0,:,:,0] - board[0,:,:,1]
    else:
        real_board = board[0,:,:,1] - board[0,:,:,0]
    return real_board

def show_board(board):
    real_board = get_real_board(board)
    for row in real_board:
        for c in row:
            if c == 1:
                print u"○",
            elif c == -1:
                print u"●",
            else:
                print u".",
        print ""

def take_stones(board):
    # TODO implement later
    return board

def make_play(x, y, board):
    player = board[0,0,0,-1]
    board[:,:,:,2:16] = board[:,:,:,0:14]
    if y != 19:
        board[0,y,x,0] = 1  # Careful here about indices
        board = take_stones(board)
    else:
        # "Skipping", player
        pass
    # swap_players
    board[:,:,:,range(16)] = board[:,:,:,SWAP_INDEX]
    player = 0 if player == 1 else 1
    board[:,:,:,-1] = player
    return board, player

def _color_adjoint(i, j, color, board):
    # TOP
    SIZE1 = len(board)
    SIZE2 = len(board[0])
    if i > 0 and board[i-1][j] == 0:
        board[i-1][j] = color
        _color_adjoint(i - 1, j, color, board)
    # BOTTOM
    if i < SIZE1 - 1 and board[i+1][j] == 0:
        board[i+1][j] = color
        _color_adjoint(i + 1, j, color, board)
    # LEFT
    if j > 0 and board[i][j - 1] == 0:
        board[i][j - 1] = color
        _color_adjoint(i, j - 1, color, board)
    # RIGHT
    if j < SIZE2 - 1 and board[i][j + 1] == 0:
        board[i][j + 1] = color
        _color_adjoint(i, j + 1, color, board)
    return board

def color_board(real_board, color):
    board = np.copy(real_board)
    for i, row in enumerate(board):
        for j, v in enumerate(row):
            if v == color:
                _color_adjoint(i, j, color, board)
    return board


def get_winner(board):
    real_board = get_real_board(board)
    points =  _get_points(real_board)
    black = points.get(1, 0) + points.get(2, 0)
    white = points.get(-1, 0) + points.get(-2, 0)
    if black > white:
        return 1, black, white
    elif black == white:
        return 0, black, white
    else:
        return -1, black, white

def _get_points(real_board):
    colored1 = color_board(real_board,  1)
    colored2 = color_board(real_board, -1)
    total = colored1 + colored2
    unique, counts = np.unique(total, return_counts=True)
    points = dict(zip(unique, counts))
    return points

def self_play_game(model, mcts_simulations):
    board = np.zeros((1, 19, 19, 17), dtype=np.float32)
    player = 1
    board[:,:,:,-1] = player
    start = datetime.datetime.now()
    skipped_last = False
    temperature = 1
    mcts_tree = None
    for i in range(22):
        if i == 30:
            temperature = 0
        policy, value = model.predict(board)
        if mcts_tree is None:
            mcts_tree = new_leaf(policy)
        index = select_play(policy, board, mcts_simulations, mcts_tree, temperature, model)
        x, y = index2coord(index)
        mcts_tree = mcts_tree[index]['subtree']
        if skipped_last and y == 19:
            break
        skipped_last = y == 19
        # print "Before play"
        # show_board(board)
        board, player = make_play(x, y, board)
        print "Player ", player, "(", x, y, ")"
        # print "Move ", i + 1, datetime.datetime.now() - start
        print len(mcts_tree)
        show_board(board)
    show_board(board)

    winner, black_points, white_points = get_winner(board)
    player_string = {1: "B", 0: "D", -1: "W"}
    winner_string = "%s+%s" % (player_string[winner], abs(black_points - white_points))
    print "Game played (%s) : %s" % (winner_string, datetime.datetime.now() - start)
    return winner


def self_play(model_name, n_games, mcts_simulations):
    model = load_model(os.path.join(conf['MODEL_DIR'], model_name), custom_objects={'loss': loss})
    for i in range(n_games):
        self_play_game(model, mcts_simulations)
