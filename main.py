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
    model_name = "model_1.h5"
    create_initial_model(name=model_name)
    self_play(model_name=model_name, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])

    model = load_latest_model()
    train(model)

    best_model = load_model(os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL']), custom_objects={'loss': loss})
    evaluate(best_model, model)

if __name__ == "__main__":
    main()
