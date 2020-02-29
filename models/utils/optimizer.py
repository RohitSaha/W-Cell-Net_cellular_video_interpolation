import tensorflow as tf
import numpy as np

def get_optimizer(train_loss, optim_id=1,
                    learning_rate=1e-3,
                    use_batch_norm=False):

    if optim_id == 1:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
    elif optim_id == 2:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9,
            use_nesterov=True)

    if use_batch_norm:
        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                train_loss)
    else:
        train_op = optimizer.minimize(
            train_loss)

    return train_op


def count_parameters():
    return np.sum(
        [np.prod(
            v.get_shape().as_list())
            for v in tf.trainable_variables()]) 


