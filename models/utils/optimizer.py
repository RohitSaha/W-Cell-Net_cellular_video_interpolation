import tensorflow as tf
import numpy as np

def get_optimizer(train_loss, optim_id=1, learning_rate=1e-3):
    if optim_id == 1:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

    train_op = optimizer.minimize(
        train_loss)

    return train_op


def count_parameters():
    return np.sum(
        [np.prod(
            v.get_shape().as_list())
            for v in tf.trainable_variables()]) 


