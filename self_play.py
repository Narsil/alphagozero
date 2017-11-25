# -*- coding: utf-8 -*-
import os
from conf import conf
import numpy as np
import numpy.ma as ma
from numpy.ma.core import MaskedConstant
import datetime
from math import sqrt
import h5py
import tqdm
from symmetry import random_symmetry_predict
from sgfsave import save_game_sgf
from play import (
    legal_moves, index2coord, make_play, game_init,
    choose_first_player,
    show_board, get_winner,game_final,get_real_board,
    decide_disable_resign,
)

SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
DIRICHLET_ALPHA = conf['DIRICHLET_ALPHA']
DIRICHLET_EPSILON = conf['DIRICHLET_EPSILON']
Cpuct = 1

def show_tree(x, y, tree, indent=''):
    print('%s Move(%s,%s) p: %s, count: %s' % (indent, x, y, tree['p'], tree['count']))
    for action, node in tree['subtree'].items():
        x, y = index2coord(action)
        show_tree(x, y, node, indent=indent+'--')


def new_tree(policy, board, add_noise=False):
    mcts_tree = {
        'count': 0,
        'value': 0,
        'mean_value': 0,
        'p': 1,
        'subtree':{},
        'parent': None,
    }
    subtree = new_subtree(policy, board, parent=mcts_tree, add_noise=add_noise)
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

def simulate(node, board, model, mcts_batch_size, original_player):
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
                    make_play(x, y, tmp_board)

                boards[i] = tmp_board
            else:
                tmp_board = np.copy(board)
                x, y = index2coord(action)
                make_play(x, y, tmp_board)
                boards[i] = tmp_board

        # The random symmetry will changes boards, so copy them before hand
        presymmetry_boards = np.copy(boards)

        policies, values = random_symmetry_predict(model, boards)

        for policy, v, board, action in zip(policies, values, presymmetry_boards, max_actions):
            shape = board.shape
            board = board.reshape([1] + list(shape))
            player = board[0,0,0,-1]
            # Inverse value if we're looking from other player perspective
            value = v[0] if player == original_player else -v[0]

            subtree = new_subtree(policy, board, node)
            leaf_node = action['node']
            leaf_node['subtree'] = subtree

            current_node = leaf_node
            while True:
                current_node['count'] += 1
                current_node['value'] += value
                current_node['mean_value'] = current_node['value'] / float(current_node['count'])
                if current_node['parent']:
                    current_node = current_node['parent']
                else:
                    break
    else:
        x, y = index2coord(selected_action)
        make_play(x, y, board)
        simulate(selected_node, board, model, mcts_batch_size, original_player)

def mcts_decision(value,policy, board, mcts_simulations, mcts_tree, temperature, model,skip_resign=False):
    for i in range(int(mcts_simulations/MCTS_BATCH_SIZE)):
        test_board = np.copy(board)
        original_player = board[0,0,0,-1]
        simulate(mcts_tree, test_board, model, MCTS_BATCH_SIZE, original_player)

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
        if skip_resign:
            _, _, selected_a = max((dic['count'], dic['mean_value'], a) 
                    for a, dic in mcts_tree['subtree'].items()
                    if a<SIZE*SIZE)
        elif value<0.05:
            selected_a=SIZE*SIZE
        else :
            _, _, selected_a = max((dic['count'], dic['mean_value'], a) for a, dic in mcts_tree['subtree'].items())
    return selected_a

def select_play(value,policy, board, mcts_simulations, mcts_tree, temperature, model,skip_resign=False):
    mask = legal_moves(board)
    policy = ma.masked_array(policy, mask=mask)
    index = mcts_decision(value,policy, board, mcts_simulations, mcts_tree,
            temperature, model,skip_resign=skip_resign)

    # index = np.argmax(policy)
    x, y = index2coord(index)
    return index

def play_game(model1, model2, mcts_simulations, stop_exploration, self_play=False, num_moves=None):
    board, player = game_init()
    moves = []

    current_model, other_model = choose_first_player(model1, model2)
    mcts_tree, other_mcts = None, None

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
    disable_resign=decide_disable_resign()
    skip_resign=False
    for move_n in range(num_moves):
        last_value = value
        if move_n == stop_exploration:
            temperature = 0
        policies, values = current_model.predict_on_batch(board)
        policy = policies[0]
        value = values[0]
        # Start of the game mcts_tree is None, but it can be {} if we selected a play that mcts never checked
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)
            if self_play:
                other_mcts = mcts_tree

        if disable_resign:
            skip_resign=not game_final(get_real_board(board))

        
        index = select_play(value,policy, board, mcts_simulations, mcts_tree, temperature, current_model,skip_resign)
        x, y = index2coord(index)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for _index, d in mcts_tree['subtree'].items():
            policy_target[_index] = d['p']

        move_data = {
            'board': np.copy(board),
            'policy': policy_target,
            'value': value,
            'move': (x, y),
            'move_n': move_n,
            'player': player ,
        }
        moves.append(move_data)

        if skipped_last and y == SIZE :
            end_reason = "BOTH_PASSED"
            break
        skipped_last = y == SIZE


        # Update trees
        if not self_play:
            # Update other only if we are not in self_play
            if other_mcts and index in other_mcts['subtree']:
                other_mcts = other_mcts['subtree'][index]
                other_mcts['parent'] = None # Cut the tree
        else:
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

            print("hand %s %s(%s,%s)" % (move_n,color, x, y)) 
            print("")
            show_board(board)
            print("")
    else:
        # When finishing due to ending the game, add the last configuration
        # to the data
        policies, values = current_model.predict_on_batch(board)
        policy = policies[0]
        value = values[0]
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)
        index = select_play(value,policy, board, mcts_simulations, 
                mcts_tree, temperature, current_model,skip_resign)
        x, y = index2coord(index)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for index, d in mcts_tree['subtree'].items():
            policy_target[index] = d['p']
        move_data = {
            'board': np.copy(board),
            'policy': policy_target,
            'value': value,
            'move': (x, y),
            'move_n': move_n,
            'player': player,
        }
        moves.append(move_data)



    winner, black_points, white_points = get_winner(board)
    player_string = {1: "B", 0: "D", -1: "W"}
    winner_string = "%s+%s" % (player_string[winner], abs(black_points - white_points))
    winner_result = {1: 1, -1: 0, 0: None}

    if winner == 0:
        winner_model = None
    else:
        winner_model = model1 if (winner == 1) == model1_isblack else model2

    if model1_isblack:
        modelB, modelW = model1, model2
    else:
        modelW, modelB = model1, model2

    if player == 0:
        # black played last
        bvalue, wvalue = value, last_value
    else:
        bvalue, wvalue = last_value, value


    if conf['SHOW_END_GAME']:
        print("")
        print("B:%s, W:%s" %(modelB.name, modelW.name))
        print("Bvalue:%s, Wvalue:%s" %(bvalue, wvalue))
        show_board(board)
        print("Game played (%s: %s) : %s" % (winner_string, end_reason, datetime.datetime.now() - start))

    game_data = {
        'moves': moves,
        'modelB_name': modelB.name,
        'modelW_name': modelW.name,
        'winner': winner_result[winner],
        'winner_model': winner_model.name,
        'result': winner_string,
    }
    return game_data


def self_play(model, n_games, mcts_simulations):
    desc = "Self play %s" % model.name
    games = tqdm.tqdm(range(n_games), desc=desc)
    games_data = []
    for game in games:
        start = datetime.datetime.now()
        game_data = play_game(model, model, mcts_simulations, conf['STOP_EXPLORATION'], self_play=True)
        stop = datetime.datetime.now()
        moves = len(game_data['moves'])
        games.set_description(desc + " %s moves %.2fs/move " % (moves, (stop - start).seconds / moves))
        save_game_data(model.name, game, game_data)
        games_data.append(game_data)
    return games_data


def save_file(model_name, game_n, move_data, winner):
    board = move_data['board']
    policy_target = move_data['policy']
    player = move_data['player']
    value_target = 1 if winner == player else -1
    move = move_data['move_n']
    directory = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d" % game_n, "move_%03d" % move)
    try:
        os.makedirs(directory)
    except OSError:
        while True:
            game_n += 1
            directory = os.path.join(conf['GAMES_DIR'], model_name, "game_%03d" % game_n, "move_%03d" % move)
            try:
                os.makedirs(directory)
                break
            except OSError:
                pass

    with h5py.File(os.path.join(directory, 'sample.h5'),'w') as f:
        f.create_dataset('board', data=board, dtype=np.float32)
        f.create_dataset('policy_target', data=policy_target, dtype=np.float32)
        f.create_dataset('value_target', data=np.array(value_target), dtype=np.float32)

def save_game_data(model_name, game_n, game_data):
    winner = game_data['winner']
    for move_data in game_data['moves']:
        save_file(model_name, game_n, move_data, winner)
    if conf['SGF_ENABLED']:
        save_game_sgf(model_name, game_n, game_data)


