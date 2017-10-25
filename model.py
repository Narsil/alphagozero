import tensorflow as tf
from conf import conf
from keras.layers import (
        Conv2D, BatchNormalization, Input, Activation, Dense, Reshape,
        Add,
)
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
import os


def residual_block(input_, node_name):
    with tf.name_scope(node_name):
        conv1 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(input_)
        batch1 = BatchNormalization()(conv1)
        relu = Activation('relu')(batch1)
        conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(relu)
        batch2 = BatchNormalization()(conv2)
        add = Add()([batch2, input_])
        out = Activation('relu')(add)
    return out


def loss(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    categorical_crossentropy = K.categorical_crossentropy(y_true, y_pred)

    regularization = K.mean(K.square(y_pred))
    epsilon = K.variable([1e-4], dtype="float32")
    c_regularization = epsilon * regularization


    total = K.concatenate([mse, categorical_crossentropy, c_regularization], axis=-1)
    total_loss = K.sum(total, axis=-1)
    return total_loss


def build_model():
    with tf.name_scope('input'):
        _input = Input(shape=(19, 19, 17))
        conv1 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                data_format='channels_last')(_input)
        batch1 = BatchNormalization()(conv1)
        relu = Activation('relu')(batch1)


    tower_input = relu
    with tf.name_scope('tower'):
        for i in range(conf['N_RESIDUAL_BLOCKS']):
            tower_output = residual_block(tower_input, node_name="residual_%s" % i)
            tower_input = tower_output



    with tf.name_scope('policy'):
        policy_conv = Conv2D(filters=2, kernel_size=(1, 1), strides=1)(tower_output)
        policy_batch = BatchNormalization()(policy_conv)
        policy_relu = Activation('relu')(policy_batch)
        policy_shape = (reduce(lambda x, y: x * y, policy_relu._keras_shape[1:]), )
        policy_reshape = Reshape(target_shape=policy_shape)(policy_relu)
        policy_out = Dense(362, activation='softmax', name="policy_out")(policy_reshape)

    with tf.name_scope('value'):
        value_conv = Conv2D(filters=2, kernel_size=(1, 1), strides=1)(tower_output)
        value_batch = BatchNormalization()(value_conv)
        value_relu = Activation('relu')(value_batch)
        value_shape = (reduce(lambda x, y: x * y, value_relu._keras_shape[1:]), )
        value_reshape = Reshape(target_shape=value_shape)(value_relu)
        value_hidden = Dense(256, activation='relu')(value_reshape)
        value_out = Dense(1, activation='tanh', name="value_out")(value_hidden)

    model = Model(inputs=[_input], outputs=[policy_out, value_out])
    sgd = SGD(lr=1e-2, momentum = 0.9)
    model.compile(sgd, loss=loss)
    return model

def create_initial_model(name):
    full_filename = os.path.join(conf['MODEL_DIR'], name)
    if os.path.isfile(full_filename):
        return
    model = build_model()
    tf_callback = TensorBoard(log_dir=conf['LOG_DIR'],
            histogram_freq=0, batch_size=1, write_graph=True, write_grads=True)
    tf_callback.set_model(model)
    tf_callback.on_epoch_end(0)
    tf_callback.on_train_end(0)
    # batch_size = 10
    # X = np.random.rand( batch_size, 19, 19, 17)
    # print model.predict(X)
    model.save(full_filename)
