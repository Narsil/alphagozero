from symmetry import random_symmetry_predict
from math import sqrt
import numpy as np
import numpy.ma as ma
from numpy.ma.core import MaskedConstant
from conf import conf
from play import (
    legal_moves, index2coord, make_play,
    coord2index,
)

SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
DIRICHLET_ALPHA = conf['DIRICHLET_ALPHA']
DIRICHLET_EPSILON = conf['DIRICHLET_EPSILON']
RESIGNATION_PERCENT = conf['RESIGNATION_PERCENT']
RESIGNATION_ALLOWED_ERROR = conf['RESIGNATION_ALLOWED_ERROR']
Cpuct = 1

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

        if len(max_actions) < top_n or v > max_actions[-1]['value']:
            max_actions.append({'action': a, 'value': v, 'node': dic})
            max_actions.sort(key=lambda x: x['value'], reverse=True)
        if len(max_actions) > top_n:
            max_actions = max_actions[:-1]
    return max_actions

def simulate(node, board, model, mcts_batch_size, original_player):
    node_subtree = node['subtree']
    max_actions = top_n_actions(node_subtree, mcts_batch_size)

    leg = legal_moves(board)
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

def mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model):
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
        _, _, selected_a = max((dic['count'], dic['mean_value'], a) for a, dic in mcts_tree['subtree'].items())
    return selected_a

def select_play(policy, board, mcts_simulations, mcts_tree, temperature, model):
    mask = legal_moves(board)
    policy = ma.masked_array(policy, mask=mask)
    index = mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model)

    x, y = index2coord(index)
    return index

class Tree(object):
    def __init__(self):
        self.tree = None

    def new_tree(self, policy, board, add_noise=False):
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
        self.tree = mcts_tree
        return mcts_tree

    def play(self, index):
        try:
            self.tree = self.tree['subtree'][index]
            self.tree['parent'] = None # Cut the tree
        except:
            self.tree = None


class ModelEngine(object):
    def __init__(self, model, mcts_simulations, board, resign=None, temperature=0, add_noise=False):
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.resign = resign
        self.temperature = temperature
        self.board = board
        self.add_noise = add_noise
        self.tree = Tree()

    def set_temperature(self, temperature):
        self.temperature = temperature

    def play(self, color, x, y):
        index = coord2index(x, y)

        self.tree.play(index)

        self.board, _ = make_play(x, y, self.board)

    def genmove(self, color):
        policies, values = self.model.predict_on_batch(self.board)
        policy = policies[0]
        value = values[0]
        if self.resign and value <= self.resign:
            x = 0
            y = SIZE + 1
            return x, y, policy, value

        # Start of the game mcts_tree is None, but it can be {} if we selected a play that mcts never checked
        if not self.tree.tree or not self.tree.tree['subtree']:
            self.tree.new_tree(policy, self.board, add_noise=self.add_noise)


        index = select_play(policy, self.board, self.mcts_simulations, self.tree.tree, self.temperature, self.model)
        x, y = index2coord(index)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for _index, d in self.tree.tree['subtree'].items():
            policy_target[_index] = d['p']

        self.board, _ = make_play(x, y, self.board)
        return x, y, policy_target, value
