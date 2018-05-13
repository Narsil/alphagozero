import os
import h5py
import numpy as np
from keras.callbacks import TensorBoard, TerminateOnNaN
from random import choices
from conf import conf
import tqdm

SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
NUM_WORKERS = conf['NUM_WORKERS']
VALIDATION_SPLIT = conf['VALIDATION_SPLIT']
MOVE_INDEX = conf['MOVE_INDEX']
GAME_FILE = conf['GAME_FILE']


def load_moves(directory):
    weights= []
    indices = []
    with open(os.path.join(directory, conf['MOVE_INDEX']), 'r') as f:
        for line in f:
            game_n, move_n, variation = line.strip().split(',')
            weights.append(float(variation))
            indices.append((int(game_n), int(move_n)))
    return indices, weights

def train(model, game_model_name, epochs=None):
    if epochs is None:
        epochs = EPOCHS_PER_SAVE
    name = model.name
    base_name, index = name.split('_')
    new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"
    tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], new_name),
            histogram_freq=conf['HISTOGRAM_FREQ'], batch_size=BATCH_SIZE, write_graph=False, write_grads=False)
    nan_callback = TerminateOnNaN()

    directory = os.path.join("games", game_model_name)
    indices, weights = load_moves(directory)
    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        for worker in tqdm.tqdm(range(NUM_WORKERS), desc="Worker_batch"):

            chosen = choices(indices, weights, k = BATCH_SIZE)

            X = np.zeros((BATCH_SIZE, SIZE, SIZE, 17))
            policy_y = np.zeros((BATCH_SIZE, SIZE*SIZE + 1))
            value_y = np.zeros((BATCH_SIZE, 1))
            for j, (game_n, move) in enumerate(chosen):
                filename = os.path.join(directory, GAME_FILE % game_n)
                with h5py.File(filename, 'r') as f:
                    board = f['move_%s/board' % move][:]
                    policy = f['move_%s/policy_target' % move][:]
                    value_target = f['move_%s/value_target' % move][()]


                    X[j] = board
                    policy_y[j] = policy
                    value_y[j] = value_target

            fake_epoch = epoch * NUM_WORKERS + worker # For tensorboard
            model.fit(X, [policy_y, value_y],
                initial_epoch=fake_epoch,
                epochs=fake_epoch + 1,
                validation_split=VALIDATION_SPLIT, # Needed for TensorBoard histograms and gradi
                callbacks=[tf_callback, nan_callback],
                verbose=0,
            )
    model.name = new_name.split('.')[0]
    model.save(os.path.join(conf['MODEL_DIR'], new_name))

