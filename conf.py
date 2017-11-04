conf = {
    'MODEL_DIR': 'models',
    'LOG_DIR': 'logs',
    'BEST_MODEL': 'best_model.h5',

    'N_RESIDUAL_BLOCKS': 20,

    'N_GAMES': 10,  # 25k in Deepmind
    'MCTS_SIMULATIONS': 2, # 1.6k in Deepmind
    'SIZE': 9, # board size
    'STOP_EXPLORATION': 30, # Number of plays after which temperature goes to 0

    'TRAIN_BATCH_SIZE': 32,
    'EPOCHS_PER_SAVE': 4, # A model will be saved to be evaluated this amount of epochs,
    'NUM_WORKERS': 2,# We use this many GPU workers so split the task,
    'L2_EPSILON': 1e-4,# The epsilon coefficient in the loss value function

    'EVALUATE_N_GAMES': 10,# The number of games to test on to elect new best model
    'EVALUATE_MARGIN': .10,# Model has to win by that margin to be elected
}
