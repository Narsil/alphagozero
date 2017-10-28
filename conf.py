conf = {
    'MODEL_DIR': 'models',
    'LOG_DIR': 'logs',
    'N_RESIDUAL_BLOCKS': 20,
    'N_GAMES': 10,  # 25k in Deepmind
    'MCTS_SIMULATIONS': 2, # 1.6k in Deepmind
    'SIZE': 9, # board size
    'STOP_EXPLORATION': 30, # Number of plays after which temperature goes to 0
    'TRAIN_BATCH_SIZE': 32,
}
