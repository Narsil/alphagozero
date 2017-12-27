# -*- coding: utf-8 -*-
import os
from conf import conf
import numpy as np
import datetime
import h5py
import tqdm
from sgfsave import save_game_sgf
from play import (
    index2coord, make_play, game_init,
    show_board, get_winner,
)
from engine import ModelEngine
from random import random

SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
DIRICHLET_ALPHA = conf['DIRICHLET_ALPHA']
DIRICHLET_EPSILON = conf['DIRICHLET_EPSILON']
RESIGNATION_PERCENT = conf['RESIGNATION_PERCENT']
RESIGNATION_ALLOWED_ERROR = conf['RESIGNATION_ALLOWED_ERROR']
Cpuct = 1

def show_tree(x, y, tree, indent=''):
    print('%s Move(%s,%s) p: %s, count: %s' % (indent, x, y, tree['p'], tree['count']))
    for action, node in tree['subtree'].items():
        x, y = index2coord(action)
        show_tree(x, y, node, indent=indent+'--')



def play_game(model1, model2, mcts_simulations, stop_exploration, self_play=False, num_moves=None, resign_model1=None, resign_model2=None):
    board, player = game_init()
    moves = []


    engine1 = ModelEngine(model1, mcts_simulations, resign=resign_model1, temperature=1, board=np.copy(board), add_noise=self_play)
    engine2 = ModelEngine(model2, mcts_simulations, resign=resign_model2, temperature=1, board=np.copy(board), add_noise=self_play)

    if self_play:
        engine2.tree = engine1.tree

    last_value = None
    value = None

    skipped_last = False
    start = datetime.datetime.now()
    end_reason = "PLAYED ALL MOVES"

    if num_moves is None:
        num_moves = SIZE * SIZE * 2

    for move_n in range(num_moves):
        last_value = value
        if move_n == stop_exploration:
            engine1.set_temperature(0)
            engine2.set_temperature(0)

        if move_n % 2 == 0:
            x, y, policy_target, value = engine1.genmove("B")
            if y == SIZE + 1:
                end_reason = 'RESIGN'
                break
            engine2.play("B", x, y, update_tree=not self_play)
        else:
            x, y, policy_target, value = engine2.genmove("W")
            if y == SIZE + 1:
                end_reason = 'RESIGN'
                break
            engine1.play("B", x, y, update_tree=not self_play)

        move_data = {
            'board': np.copy(board),
            'policy': policy_target,
            'value': value,
            'move': (x, y),
            'move_n': move_n,
            'player': player ,
        }
        moves.append(move_data)

        if skipped_last and y == SIZE:
            end_reason = "BOTH_PASSED"
            break
        skipped_last = y == SIZE

        if y == SIZE + 1:
            end_reason = 'RESIGN'
            break

        # Swap players
        board, player = make_play(x, y, board)

        if conf['SHOW_EACH_MOVE']:
            # Inverted here because we already swapped players
            color = "W" if player == 1 else "B"

            print("%s(%s,%s)" % (color, x, y)) 
            print("")
            print(show_board(board))
            print("")


    winner, black_points, white_points = get_winner(board)
    player_string = {1: "B", 0: "D", -1: "W"}
    if end_reason == "resign":
        winner_string = "%s+R" % (player_string[player])
    else:
        winner_string = "%s+%s" % (player_string[winner], abs(black_points - white_points))

    winner_engine = engine1 if (winner == 1) else engine2
    modelB, modelW = model1, model2



    if conf['SHOW_END_GAME']:
        if player == 0:
            # black played last
            bvalue, wvalue = value, last_value
        else:
            bvalue, wvalue = last_value, value
        print("")
        print("B:%s, W:%s" %(modelB.name, modelW.name))
        print("Bvalue:%s, Wvalue:%s" %(bvalue, wvalue))
        print(show_board(board))
        print("Game played (%s: %s) : %s" % (winner_string, end_reason, datetime.datetime.now() - start))

    game_data = {
        'moves': moves,
        'modelB_name': modelB.name,
        'modelW_name': modelW.name,
        'winner': winner,
        'winner_model': winner_engine.model.name,
        'result': winner_string,
        'resign_model1': resign_model1,
        'resign_model2': resign_model2,
    }
    return game_data


def self_play(model, n_games, mcts_simulations):
    desc = "Self play %s" % model.name
    games = tqdm.tqdm(range(n_games), desc=desc)
    games_data = []
    current_resign = None
    min_values = []
    for game in games:

        if random() > RESIGNATION_PERCENT:
            resign = current_resign
        else:
            resign = None

        start = datetime.datetime.now()
        game_data = play_game(model, model, mcts_simulations, conf['STOP_EXPLORATION'], self_play=True, resign_model1=resign, resign_model2=resign)
        stop = datetime.datetime.now()

        # If we did not use resignation, we had the result towards resign value.
        if resign == None:
            winner = game_data['winner']
            if winner == 1:
                min_value = min([move['value'] for move in game_data['moves'][::2]])
            else:
                min_value = min([move['value'] for move in game_data['moves'][1::2]])
            min_values.append(min_value)
            l = len(min_values)
            resignation_index = int(RESIGNATION_ALLOWED_ERROR * l)
            if resignation_index > 0:
                current_resign = min_values[resignation_index]

        moves = len(game_data['moves'])
        speed = ((stop - start).seconds / moves) if moves else 0.
        games.set_description(desc + " %s moves %.2fs/move " % (moves, speed))
        save_game_data(model.name, game_data)
        games_data.append(game_data)
    return games_data

def get_game_n(model_name):
    directory = os.path.join(conf["GAMES_DIR"], model_name)
    try:
        os.makedirs(directory)
    except:
        pass
    dirs = os.listdir(directory)
    index = [int(name.split("_")[-1].split('.')[0]) for name in dirs] # game_001, and game_001.sgf
    return max(index, default=0) + 1

def save_file(model_name, game_n, move_data, winner):
    board = move_data['board']
    policy_target = move_data['policy']
    player = move_data['player']
    value_target = 1 if winner == player else -1
    move = move_data['move_n']
    directory = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d" % game_n, "move_%03d" % move)
    os.makedirs(directory)
    with h5py.File(os.path.join(directory, 'sample.h5'),'w') as f:
        f.create_dataset('board', data=board, dtype=np.float32)
        f.create_dataset('policy_target', data=policy_target, dtype=np.float32)
        f.create_dataset('value_target', data=np.array(value_target), dtype=np.float32)

def save_game_data(model_name, game_data):
    winner = game_data['winner']
    game_n = get_game_n(model_name)
    for move_data in game_data['moves']:
        save_file(model_name, game_n, move_data, winner)
    if conf['SGF_ENABLED']:
        save_game_sgf(model_name, game_n, game_data)


