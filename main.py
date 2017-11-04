import os
from model import create_initial_model, loss, load_latest_model
from keras.models import load_model
from self_play import self_play
from train import train
from conf import conf
from evaluator import evaluate


def init_directories():
    try:
        os.mkdir(conf['MODEL_DIR'])
    except:
        pass
    try:
        os.mkdir(conf['MODEL_DIR'])
    except:
        pass

def main():

    init_directories()
    model_name = "model_1"
    model = create_initial_model(name=model_name)
    print("Created random model")
    self_play(model, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])
    print("Played %s games with model %s" % (conf['N_GAMES'], model_name))

    while True:
        model = load_latest_model()
        best_model = load_model(os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL']), custom_objects={'loss': loss})
        train(model, game_model_name=best_model.name)
        print("Trained model")
        new_is_best = evaluate(best_model, model)
        if new_is_best:
            self_play(model, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])

if __name__ == "__main__":
    main()
