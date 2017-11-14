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
import h5py
from random import random
import tqdm
from symmetry import random_symmetry_predict

SWAP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
DIRICHLET_ALPHA = conf['DIRICHLET_ALPHA']
DIRICHLET_EPSILON = conf['DIRICHLET_EPSILON']
Cpuct = 1

def index2coord(index):
    y = index // SIZE
    x = index - SIZE * y
    return x, y

def legal_moves(board):
    # Occupied places
    mask1 = board[0,:,:,0].reshape(-1) != 0
    mask2 = board[0,:,:,1].reshape(-1) != 0
    mask = mask1 + mask2

    # Ko situations
    ko_mask = (board[0,:,:,2] - board[0,:,:,0])
    if (ko_mask == 1).sum() == 1:
        mask += (ko_mask == 1).reshape(-1)

    # Pass is always legal
    mask = np.append(mask, 0)
    return mask

def new_tree(policy, board, add_noise=False):
    mcts_tree = {
        'count': 0,
        'value': 0,
        'mean_value': 0,
        'p': 1,
        'subtree':{},
        'parent': None,
    }
    subtree = new_subtree(policy, board, parent=mcts_tree, add_noise=self_play)
    mcts_tree['subtree'] = subtree
    return mcts_tree

def new_subtree(policy, board, parent, add_noise=False):
    leaf = {}

    # We need to check for legal moves here because MCTS might not have expanded
    # this subtree
    mask = legal_moves(board)
    policy = ma.masked_array(policy, mask=mask)

    # Add Dirichlet noise.
    tmp = policy.reshape(-1)
    if add_noise:
        noise = np.random.dirichlet([DIRICHLET_ALPHA for i in range(tmp.shape[0])])
        tmp = (1 - DIRICHLET_EPSILON) * tmp + DIRICHLET_EPSILON * noise


    for move, p in enumerate(tmp):
        if isinstance(p, MaskedConstant):
            continue

        leaf[move] = {
            'count': 0,
            'value': 0,
            'mean_value': 0,
            'p': p,
            'subtree':{},
            'parent': parent,
        }
    return leaf

def top_n_actions(subtree, top_n):
    total_n = sqrt(sum(dic['count'] for dic in subtree.values()))
    if total_n == 0:
        total_n = 1
    # Select exploration
    max_actions = []
    for a, dic in subtree.items():
        u = Cpuct * dic['p'] * total_n / (1. + dic['count']) 
        v = dic['mean_value'] + u

        if len(max_actions) < top_n or v > max_actions[0]['value']:
            max_actions.append({'action': a, 'value': v, 'node': dic})
            max_actions.sort(key=lambda x: x['value'], reverse=True)
        if len(max_actions) > top_n:
            max_actions = max_actions[:-1]
    return max_actions

def simulate(node, board, model, mcts_batch_size):
    node_subtree = node['subtree']
    max_actions = top_n_actions(node_subtree, mcts_batch_size)
    max_a = max_actions[0]['action']

    selected_action = max_a
    selected_node = node_subtree[selected_action]
    if selected_node['subtree'] == {}:
        # This is a leaf
        boards = np.zeros((mcts_batch_size, SIZE, SIZE, 17), dtype=np.float32)
        for i, dic in enumerate(max_actions):
            action = dic['action']
            if dic['node']['subtree'] != {}:
                # already expanded
                tmp_node = dic['node']
                tmp_action = action
                tmp_board = np.copy(board)
                x, y = index2coord(tmp_action)
                tmp_board, _ = make_play(x, y, tmp_board)
                while tmp_node['subtree'] != {}:
                    tmp_max_actions = top_n_actions(tmp_node['subtree'], mcts_batch_size)
                    tmp_d = tmp_max_actions[0]
                    tmp_node = tmp_d['node']
                    tmp_action = tmp_d['action']
                    # The node for this action is the leaf, this is where the
                    # update will start, working up the tree
                    dic['node'] = tmp_node 
                    x, y = index2coord(tmp_action)
                    tmp_board, _ = make_play(x, y, tmp_board)
                boards[i] = tmp_board
            else:
                tmp_board = np.copy(board)
                x, y = index2coord(action)
                player = tmp_board[0,0,0,-1]
                tmp_board, player = make_play(x, y, tmp_board)
                boards[i] = tmp_board

        policies, values = random_symmetry_predict(model, boards)
        for policy, value, board, action in zip(policies, values, boards, max_actions):
            shape = board.shape
            board = board.reshape([1] + list(shape))
            mask = legal_moves(board)
            policy = ma.masked_array(policy, mask=mask)
            subtree = new_subtree(policy, board, node)
            leaf_node = action['node']
            leaf_node['subtree'] = subtree

            current_node = leaf_node
            while True:
                current_node['count'] += 1
                current_node['value'] += value[0]
                current_node['mean_value'] = current_node['value'] / float(current_node['count'])
                if current_node['parent']:
                    current_node = current_node['parent']
                else:
                    break
    else:
        x, y = index2coord(selected_action)
        player = board[0,0,0,-1]
        board, player = make_play(x, y, board)
        simulate(selected_node, board, model, mcts_batch_size)

def mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model):
    for i in range(int(mcts_simulations/MCTS_BATCH_SIZE)):
        test_board = np.copy(board)
        simulate(mcts_tree, test_board, model, MCTS_BATCH_SIZE)

    if temperature == 1:
        total_n = sum(dic['count'] for dic in mcts_tree['subtree'].values())
        moves = []
        ps = []
        for move, dic in mcts_tree['subtree'].items():
            n = dic['count']
            if not n:
                continue
            p = dic['count'] / float(total_n)
            moves.append(move)
            ps.append(p)
        selected_a = np.random.choice(moves, size=1, p=ps)[0]
    elif temperature == 0:
        _, selected_a = max((dic['count'], a) for a, dic in mcts_tree['subtree'].items())
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
                print(u"○", end=' ')
            elif c == -1:
                print(u"●", end=' ')
            else:
                print(u".", end=' ')
        print("")

dxdys = [(1, 0), (-1, 0), (0, 1), (0, -1)]
def capture_group(x, y, real_board, group=None):
    if group is None:
        group = [(x, y)]

    c = real_board[y][x]
    for dx, dy in dxdys:
        nx = x + dx
        ny = y + dy
        if (nx, ny) in group:
            continue
        if not(0 <= nx < SIZE and 0 <= ny < SIZE):
            continue
        dc = real_board[ny][nx]
        if dc == 0:
            return None
        elif dc == c:
            group.append( (nx, ny) )
            group = capture_group(nx, ny, real_board, group=group)
            if group == None:
                return None
    return group

def take_stones(x, y, board):
    real_board = get_real_board(board)
    _player = 1 if board[0,0,0,-1] == 1 else -1
    for dx, dy in dxdys:  # We need to check capture
        nx = x + dx
        ny = y + dy
        if not(0 <= nx < SIZE and 0 <= ny < SIZE):
            continue
        if real_board[ny][nx] == 0:
            continue
        if real_board[ny][nx] == _player:
            continue
        group = capture_group(nx, ny, real_board)
        if group:
            for _x, _y in group:
                assert board[0,_y,_x,1] == 1
                board[0,_y,_x,1] = 0
                real_board[_y][_x] = 0
    for dx, dy in dxdys + [(0, 0)]:  # We need to check self sucide.
        nx = x + dx
        ny = y + dy
        if not(0 <= nx < SIZE and 0 <= ny < SIZE):
            continue
        if real_board[ny][nx] == 0:
            continue
        if real_board[ny][nx] != _player:
            continue
        group = capture_group(nx, ny, real_board)
        if group:
            for _x, _y in group:
                # Sucide
                assert board[0,_y,_x,0] == 1
                board[0,_y,_x,0] = 0
                real_board[_y][_x] = 0




    return board

def make_play(x, y, board):
    player = board[0,0,0,-1]
    board[:,:,:,2:16] = board[:,:,:,0:14]
    if y != SIZE:
        board[0,y,x,0] = 1  # Careful here about indices
        board = take_stones(x, y, board)
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
    white = points.get(-1, 0) + points.get(-2, 0) + conf['KOMI']
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


def game_init():
    board = np.zeros((1, SIZE, SIZE, 17), dtype=np.float32)
    player = 1
    board[:,:,:,-1] = player
    return board, player

def choose_first_player(model1, model2):
    if random() < .5:
        current_model, other_model = model1, model2
    else:
        other_model, current_model = model1, model2
    return current_model, other_model

def play_game(model1, model2, mcts_simulations, stop_exploration, self_play=False, num_moves=None):
    board, player = game_init()
    boards_and_policies = []

    current_model, other_model = choose_first_player(model1, model2)
    mcts_tree, other_mcts = None, None

    if self_play:
        other_mcts = mcts_tree  # Fuse the trees
    last_value = None
    value = None

    model1_isblack = current_model == model1

    start = datetime.datetime.now()
    skipped_last = False
    temperature = 1
    start = datetime.datetime.now()
    end_reason = "PLAYED ALL MOVES"
    if num_moves is None:
        num_moves = SIZE * SIZE * 2
    for i in range(num_moves):
        last_value = value
        if i == stop_exploration:
            temperature = 0
        policy, value = current_model.predict(board)
        # Start of the game mcts_tree is None, but it can be {} if we selected a play that mcts never checked
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)


        index = select_play(policy, board, mcts_simulations, mcts_tree, temperature, current_model)
        x, y = index2coord(index)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for index, d in mcts_tree['subtree'].items():
            policy_target[index] = d['p']

        boards_and_policies.append( (np.copy(board), policy_target) )

        if skipped_last and y == SIZE:
            end_reason = "BOTH_PASSED"
            break
        skipped_last = y == SIZE


        # Update trees
        if not self_play:
            # Update other only if we are not in self_play
            if other_mcts and index in other_mcts['subtree']:
                other_mcts = other_mcts['subtree'][index]
                other_mcts['parent'] = None # Cut the tree
        mcts_tree = mcts_tree['subtree'][index]
        mcts_tree['parent'] = None # Cut the tree

        # Swap players
        board, player = make_play(x, y, board)
        current_model, other_model = other_model, current_model
        mcts_tree, other_mcts = other_mcts, mcts_tree

        if conf['SHOW_EACH_MOVE']:
            # Inverted here because we already swapped players
            color = "W" if player == 1 else "B"

            print("%s(%s,%s)" % (color, x, y)) 
            print("")
            show_board(board)
            print("")
    else:
        # When finishing due to ending the game, add the last configuration
        # to the data
        policy, value = current_model.predict(board)
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)
        index = select_play(policy, board, mcts_simulations, mcts_tree, temperature, current_model)
        x, y = index2coord(index)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for index, d in mcts_tree['subtree'].items():
            policy_target[index] = d['p']
        boards_and_policies.append( (np.copy(board), policy_target) )



    winner, black_points, white_points = get_winner(board)
    player_string = {1: "B", 0: "D", -1: "W"}
    winner_string = "%s+%s" % (player_string[winner], abs(black_points - white_points))
    winner_result = {1: 1, -1: 0, 0: None}

    if winner == 0:
        winner_model = None
    else:
        winner_model = model1 if (winner == 1) == model1_isblack else model2

    if conf['SHOW_END_GAME']:
        if model1_isblack:
            modelB, modelW = model1, model2
        else:
            modelW, modelB = model1, model2

        if player == 0:
            # black played last
            bvalue, wvalue = value, last_value
        else:
            bvalue, wvalue = last_value, value

        print("")
        print("B:%s, W:%s" %(modelB.name, modelW.name))
        print("Bvalue:%s, Wvalue:%s" %(bvalue, wvalue))
        show_board(board)
        print("Game played (%s: %s) : %s" % (winner_string, end_reason, datetime.datetime.now() - start))
    return boards_and_policies, winner_result[winner], winner_model


def self_play(model, n_games, mcts_simulations):
    desc = "Self play %s" % model.name
    games = tqdm.tqdm(range(n_games), desc=desc)
    for game in games:
        start = datetime.datetime.now()
        boards_and_policies, winner, _ = play_game(model, model, mcts_simulations, conf['STOP_EXPLORATION'], self_play=True)
        stop = datetime.datetime.now()
        moves = len(boards_and_policies)
        games.set_description(desc + " %.2fs/move" % ( (stop - start).seconds / moves))
        if winner is None:
            continue
        for move, (board, policy_target) in enumerate(boards_and_policies):
            player = board[0,0,0,-1]
            value_target = 1 if winner == player else -1
            save_file(model, game, move, board, policy_target, value_target)

def save_file(model, game, move, board, policy_target, value_target):
    directory = os.path.join("games", model.name, "game_%03d" % game, "move_%03d" % move)
    try:
        os.makedirs(directory)
    except OSError:
        while True:
            game += 1
            directory = os.path.join("games", model.name, "game_%03d" % game, "move_%03d" % move)
            try:
                os.makedirs(directory)
                break
            except OSError:
                pass

    with h5py.File(os.path.join(directory, 'sample.h5'),'w') as f:
        f.create_dataset('board', data=board, dtype=np.float32)
        f.create_dataset('policy_target', data=policy_target, dtype=np.float32)
        f.create_dataset('value_target', data=np.array(value_target), dtype=np.float32)
