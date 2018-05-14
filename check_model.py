import tensorflow as tf
from keras import backend as K
import os
import unittest
import numpy as np
import numpy.ma as ma
from keras.models import load_model
from self_play import self_play
from engine import select_play, Tree
from model import create_initial_model, load_best_model, loss, build_model
from play import game_init, make_play, legal_moves
from train import train
from conf import conf
from main import init_directories
from evaluator import evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SIZE = conf['SIZE']
PASS = SIZE * SIZE

def give_two_eyes(board, color):
    if not color in 'BW':
        raise Exception("Invalid color")
    eyes = [[0, 0], [2, 0]] # The 2 eyes
    for i in range(SIZE):
        for j in range(SIZE):
            if color == 'W':
                make_play(0, SIZE, board)  # Black pass

            if [i, j] in eyes:
                make_play(0, SIZE, board)  # pass on two intersection
            else:
                make_play(i, j, board) 

            if color == 'B':
                make_play(0, SIZE, board) # White pass
    if color == 'W':
        make_play(0, SIZE, board) # Black last pass

class TestModelLearningTestCase(unittest.TestCase):
    def setUp(self):
        init_directories()
        model_name = "model_1"
        best_model = load_best_model()
        self.model = best_model
        self.board, player = game_init()

    def test_learned_to_pass_black(self):
        model = self.model
        board = self.board

        give_two_eyes(board, 'B')

        policies, values = model.predict_on_batch(board)
        value_target = 1. # Black should win

        self.assertLess(abs(value_target - values[0][0]), .1)
        self.assertEqual(np.argmax(policies[0]), PASS) # Pass move is best option

    def test_learned_to_pass_white(self):
        model = self.model
        board = self.board

        give_two_eyes(board, 'W')

        policies, values = model.predict_on_batch(board)
        value_target = 1. # White should win
        self.assertLess(abs(value_target - values[0][0]), .1)
        self.assertEqual(np.argmax(policies[0]), PASS) # Pass move is best option


    def test_simulation_can_recover_from_sucide_move_black(self):
        model = self.model
        board = self.board

        give_two_eyes(board, 'B')

        policies, values = model.predict_on_batch(board)
        policy = policies[0]
        if np.argmax(policy) == PASS:
            policy[0], policy[PASS] = policy[PASS], policy[0] # Make best move sucide
            mask = legal_moves(board)
            policy = ma.masked_array(policy, mask=mask)
            self.assertEqual(np.argmax(policy), 0) # Best option in policy is sucide
        else:
            print("Warning, policy is not great")

        self.assertEqual(np.argmax(policy), 0) # Best option in policy is sucide
        tree = Tree()
        tree.new_tree(policy, board)
        chosen_play = select_play(policy, board, mcts_simulations=128, mcts_tree=tree.tree, temperature=0, model=model)
        
        # First simulation chooses pass, second simulation chooses sucide (p is still higher),
        # then going deeper it chooses pass again (value is higher)
        self.assertEqual(chosen_play, PASS) # Pass move is best option

    def test_simulation_can_recover_from_sucide_move_white(self):
        model = self.model
        board, player = game_init()

        give_two_eyes(board, 'W')

        policies, values = model.predict_on_batch(board)
        policy = policies[0]

        if np.argmax(policy) == PASS:
            policy[0], policy[PASS] = policy[PASS], policy[0] # Make best move sucide
            mask = legal_moves(board)
            policy = ma.masked_array(policy, mask=mask)
            self.assertEqual(np.argmax(policy), 0) # Best option in policy is sucide
        else:
            print("Warning, policy is not great")

        tree = Tree()
        tree.new_tree(policy, board, move=2)
        chosen_play = select_play(policy, board, mcts_simulations=128, mcts_tree=tree.tree, temperature=0, model=model)
        
        # First simulation chooses pass, second simulation chooses sucide (p is still higher),
        # then going deeper it chooses pass again (value is higher)
        self.assertEqual(chosen_play, PASS) # Pass move is best option


    def test_model_learning(self):
        model = self.model
        board, player = game_init()
        for i in range(SIZE):
            for j in range(SIZE):
                if (i + j) % 2 == 0:
                    make_play(i, j, board) 
                    make_play(0, SIZE, board) # White does not play playing

        
        # Black board, black to play
        policies, values = model.predict_on_batch(board)
        self.assertGreater(values[0][0], 0.9)
        # White board, white to play
        board[:,:,:,-1] = 0
        policies, values = model.predict_on_batch(board)
        self.assertGreater(values[0][0], 0.9)

        board[:,:,:,-1] = 1
        make_play(0, SIZE, board) # black passes
        # Black board, white to play
        policies, values = model.predict_on_batch(board)
        self.assertLess(values[0][0], -0.9)

        board[:,:,:,-1] = 1
        # White board, black to play
        policies, values = model.predict_on_batch(board)
        self.assertLess(values[0][0], -0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
