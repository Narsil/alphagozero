#######################
## Snippet from https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
#######################
import numpy as np
import tensorflow as tf
from random import seed, randrange

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(0)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

seed(0)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(0)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#######################
## End of snippet
#######################



from conf import conf
conf['SIZE'] = 9  # Override settings for tests
conf['KOMI'] = 5.5  # Override settings for tests
conf['MCTS_BATCH_SIZE'] = 8  # Override settings for tests
conf['MCTS_SIMULATION'] = 32  # Override settings for tests
conf['N_GAMES'] = 100  # Override settings for tests
conf['EVALUATE_N_GAMES'] = 10  # Override settings for tests
conf['EPOCHS_PER_SAVE'] = 100  # Override settings for tests
SIZE = conf['SIZE']
import os
import unittest
import numpy as np
import numpy.ma as ma
from keras.models import load_model
from self_play import self_play, new_tree, select_play
from model import create_initial_model, load_best_model, loss, build_model
from play import game_init, make_play, legal_moves
from train import train
from conf import conf
from main import init_directories
from evaluator import evaluate


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


class SelfPlayTestCase(unittest.TestCase):
    def test_self_play(self):
        model = DummyModel()
        self.count = 0
        fn = model.predict_on_batch
        def monkey_patch(this, *args, **kwargs):
            self.count += 1
            return fn(this, *args, **kwargs)
        model.predict_on_batch = monkey_patch
            
        # mcts batch size is 8 and we need at least one batch
        games_data = self_play(model, n_games=2, mcts_simulations=8)

        self.assertEqual(len(games_data), 2)
        moves = len(games_data[0]['moves']) + len(games_data[1]['moves'])

        self.assertEqual(self.count, 2 * moves) # 1 prediction for mcts simulation + 1 to get policy


        self.count = 0
        games_data = self_play(model, n_games=2, mcts_simulations=32)

        self.assertEqual(len(games_data), 2)
        moves = len(games_data[0]['moves']) + len(games_data[1]['moves'])

        self.assertEqual(self.count, 5 * moves) # 4 predictions for mcts simulation + 1 to get policy

    def test_self_play_resign(self):
        model = DummyModel()
        games_data = self_play(model, n_games=50, mcts_simulations=8)

        self.assertEqual(len(games_data), 50)

        resign_games = len([g for g in games_data if g['resign_model1'] != None and g['resign_model2'] != None])
        no_resign_games = len([g for g in games_data if g['resign_model1'] == None or g['resign_model2'] == None])
        self.assertEqual(resign_games, 28)
        self.assertEqual(no_resign_games, 22)


class TestModelSavingTestCase(unittest.TestCase):
    def test_model_saving(self):
        init_directories()
        model_name = "model_1"
        model = build_model(model_name)

        board, player = game_init()
        policies, values = model.predict(board)
        try:
            os.remove('test.h5')
        except:
            pass
        model.save('test.h5')
        model2 = load_model('test.h5', custom_objects={'loss': loss})
        policies2, values2 = model2.predict(board)

        self.assertTrue(np.array_equal(values, values2))
        self.assertTrue(np.array_equal(policies, policies2))
        os.remove('test.h5')

    def test_model_saving_after_training(self):
        init_directories()
        model_name = "model_1"
        model = build_model(model_name)
        self.assertEqual(model.name, 'model_1')
        board, player = game_init()
        policies, values = model.predict(board)
        try:
            os.remove('test.h5')
        except:
            pass
        model.save('test.h5')
        self_play(model, n_games=2, mcts_simulations=32)
        train(model, game_model_name=model.name, epochs=2)
        self.assertEqual(model.name, 'model_2')
        policies2, values2 = model.predict(board)
        self.assertFalse(np.array_equal(values, values2))
        self.assertFalse(np.array_equal(policies, policies2))

        model3 = load_model('test.h5', custom_objects={'loss': loss})
        policies3, values3 = model3.predict(board)

        self.assertTrue(np.array_equal(values, values3))
        self.assertTrue(np.array_equal(policies, policies3))
        os.remove('test.h5')


class TestModelLearningTestCase(unittest.TestCase):
    def setUp(self):
        init_directories()
        model_name = "model_1"
        model = create_initial_model(name=model_name)
        best_model = load_best_model()
        if best_model.name == model.name:
            train(model, game_model_name=best_model.name)
            evaluate(best_model, model)
            # We save wether or not it was a better model
            full_filename = os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL'])
            model.save(full_filename)
        else:
            model = best_model
        self.model = model

    def test_learned_to_pass_black(self):
        model = self.model
        
        board, player = game_init()

        x = randrange(SIZE)
        y = randrange(SIZE)
        for i in range(SIZE):
            for j in range(SIZE):
                if i == x and j == y:
                    make_play(0, SIZE, board)  # pass on one intersection
                else:
                    make_play(i, j, board) 
                make_play(0, SIZE, board) # White does not play playing

        policies, values = model.predict_on_batch(board)
        self.assertEqual(np.argmax(policies[0]), SIZE*SIZE) # Pass move is best option

    def test_learned_to_pass_white(self):
        model = self.model
        
        board, player = game_init()

        x = randrange(SIZE)
        y = randrange(SIZE)
        for i in range(SIZE):
            for j in range(SIZE):
                make_play(0, SIZE, board) # Black does not play playing
                if i == x and j == y:
                    make_play(0, SIZE, board)  # pass on one intersection
                else:
                    make_play(i, j, board) 

        policies, values = model.predict_on_batch(board)
        policy = policies[0]
        self.assertEqual(np.argmax(policy), SIZE*SIZE) # Pass move is best option


    def test_simulation_can_recover_from_sucide_move_black(self):
        model = self.model
        board, player = game_init()

        x = randrange(SIZE)
        y = randrange(SIZE)
        for i in range(SIZE):
            for j in range(SIZE):
                if i == x and j == y:
                    make_play(0, SIZE, board)  # pass on one intersection
                else:
                    make_play(i, j, board) 
                make_play(0, SIZE, board) # White does not play playing

        policies, values = model.predict_on_batch(board)
        policy = policies[0]
        policy[y * SIZE + x], policy[SIZE * SIZE] = policy[SIZE * SIZE], policy[y * SIZE + x] # Make best move sucide
        self.assertEqual(np.argmax(policy), y * SIZE + x) # Best option in policy is sucide
        tree = new_tree(policy, board)
        chosen_play = select_play(policy, board, mcts_simulations=128, mcts_tree=tree, temperature=0, model=model)
        
        # First simulation chooses pass, second simulation chooses sucide (p is still higher),
        # then going deeper it chooses pass again (value is higher)
        self.assertEqual(chosen_play, SIZE*SIZE) # Pass move is best option

    def test_simulation_can_recover_from_sucide_move_white(self):
        model = self.model
        board, player = game_init()

        x = randrange(SIZE)
        y = randrange(SIZE)
        for i in range(SIZE):
            for j in range(SIZE):
                make_play(0, SIZE, board) # Black does not play playing
                if i == x and j == y:
                    make_play(0, SIZE, board)  # pass on one intersection
                else:
                    make_play(i, j, board) 
        make_play(0, SIZE, board) # Black does not play playing

        policies, values = model.predict_on_batch(board)
        policy = policies[0]
        policy[y * SIZE + x], policy[SIZE * SIZE] = policy[SIZE * SIZE], policy[y * SIZE + x] # Make best move sucide
        mask = legal_moves(board)
        policy = ma.masked_array(policy, mask=mask)
        self.assertEqual(np.argmax(policy), y * SIZE + x) # Best option in policy is sucide
        tree = new_tree(policy, board)
        chosen_play = select_play(policy, board, mcts_simulations=128, mcts_tree=tree, temperature=0, model=model)
        
        # First simulation chooses pass, second simulation chooses sucide (p is still higher),
        # then going deeper it chooses pass again (value is higher)
        self.assertEqual(chosen_play, SIZE*SIZE) # Pass move is best option


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
    unittest.main()
