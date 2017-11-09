import os
import h5py
import numpy as np
from keras.callbacks import TensorBoard, TerminateOnNaN
from random import sample
from conf import conf
import tqdm

SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
NUM_WORKERS = conf['NUM_WORKERS']
VALIDATION_SPLIT = conf['VALIDATION_SPLIT']

def train(model, game_model_name):
    name = model.name
    base_name, index = name.split('_')
    new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"
    tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], new_name),
            histogram_freq=conf['HISTOGRAM_FREQ'], batch_size=BATCH_SIZE, write_graph=False, write_grads=False)
    nan_callback = TerminateOnNaN()

    directory = os.path.join("games", game_model_name)
    all_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            full_filename = os.path.join(root, f)
            all_files.append(full_filename)
    for epoch in tqdm.tqdm(range(EPOCHS_PER_SAVE), desc="Epochs"):
        for worker in tqdm.tqdm(range(NUM_WORKERS), desc="Worker_batch"):
            files = sample(all_files, BATCH_SIZE)

            X = np.zeros((BATCH_SIZE, SIZE, SIZE, 17))
            policy_y = np.zeros((BATCH_SIZE, 1))
            value_y = np.zeros((BATCH_SIZE, SIZE*SIZE + 1))
            for j, filename in enumerate(files):
                with h5py.File(filename) as f:
                    board = f['board'][:]
                    policy = f['policy_target'][:]
                    value_target = f['value_target']

                    X[j] = board
                    policy_y[j] = value_target
                    value_y[j] = policy

            fake_epoch = epoch * NUM_WORKERS + worker # For tensorboard
            model.fit(X, [value_y, policy_y],
                initial_epoch=fake_epoch,
                epochs=fake_epoch + 1,
                validation_split=VALIDATION_SPLIT, # Needed for TensorBoard histograms and gradi
                callbacks=[tf_callback, nan_callback],
                verbose=0,
            )
    model.name = new_name.split('.')[0]
    model.save(os.path.join(conf['MODEL_DIR'], new_name))

