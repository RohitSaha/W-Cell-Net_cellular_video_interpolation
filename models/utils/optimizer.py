import tensorflow as tf

def get_optimizer(train_loss, optim_id=1, learning_rate=1e-3):
    if optim_id == 1:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

    train_op = optimizer.minimize(
        train_loss)

    return train_op
