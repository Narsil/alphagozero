import os
from model import create_initial_model, loss, load_latest_model
from keras.models import load_model
from keras import backend as K
from train import train
from conf import conf
from evaluator import evaluate


def init_directories():
    try:
        os.mkdir(conf['MODEL_DIR'])
    except:
        pass
    try:
        os.mkdir(conf['LOG_DIR'])
    except:
        pass

def main():

    init_directories()
    model_name = "model_1"
    model = create_initial_model(name=model_name)

    while True:
        model = load_latest_model()
        best_model = load_model(os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL']), custom_objects={'loss': loss})
        train(model, game_model_name=best_model.name)
        evaluate(best_model, model)
        K.clear_session()

if __name__ == "__main__":
    main()
