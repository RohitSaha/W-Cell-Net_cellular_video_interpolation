import tensorflow as tf

def huber_loss(prediction, ground_truth,
                delta=1.0):

    # set a small delta value to make it behave like
    # L1 loss
    loss = tf.keras.losses.Huber(delta=delta)

    return loss(
        prediction,
        ground_truth)


def l2_loss(prediction, ground_truth):

    loss = tf.nn.l2_loss(
        prediction - ground_truth)

    return loss


def ridge_weight_decay(parameters):

    weight_decay = tf.add_n(
        [
            tf.nn.l2_loss(param)
            for param in parameters
            if 'w:' in param.name \
            or 'kernel:' in param.name])
    
    return weight_decay
