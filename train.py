import os
import h5py
import numpy as np
from random import sample
from conf import conf

SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']

def train(model):
    directory = os.path.join("games", model.name)


    all_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            full_filename = os.path.join(root, f)
            all_files.append(full_filename)


    files = sample(all_files, BATCH_SIZE)
    X = np.zeros((BATCH_SIZE, SIZE, SIZE, 17))
    policy_y = np.zeros((BATCH_SIZE, 1))
    value_y = np.zeros((BATCH_SIZE, SIZE*SIZE + 1))
    for i, filename in enumerate(files):
        with h5py.File(filename) as f:
            board = f['board'][:]
            policy = f['policy_target'][:]
            value_target = f['value_target']

            X[i] = board
            policy_y[i] = value_target
            value_y[i] = policy

    model.fit(X, [value_y, policy_y])
