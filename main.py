from model import create_initial_model
from self_play import self_play
from conf import conf

def main():
    model_name = "initial_model.h5"
    create_initial_model(name=model_name)
    self_play(model_name=model_name, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])

if __name__ == "__main__":
    main()
