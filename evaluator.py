from self_play import play_game
from conf import conf
import os
import tqdm

MCTS_SIMULATIONS = conf['MCTS_SIMULATIONS']
EVALUATE_N_GAMES = conf['EVALUATE_N_GAMES']
EVALUATE_MARGIN = conf['EVALUATE_MARGIN']

def elect_model_as_best_model(model):
    full_filename = os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL'])
    model.save(full_filename)

def evaluate(best_model, tested_model):
    total = 0
    wins = 0
    for i in tqdm.tqdm(range(EVALUATE_N_GAMES), "Evaluation %s vs %s" % (best_model.name, tested_model.name)):
        _, _, winner_model = play_game(best_model, tested_model, MCTS_SIMULATIONS, stop_exploration=2)
        if winner_model == tested_model:
            wins += 1
        total += 1
        print("NEW_MODEL", wins/total)
    if wins/total > EVALUATE_MARGIN:
        print("We found a new best model !")
        elect_model_as_best_model(tested_model)
        return True
    return False
