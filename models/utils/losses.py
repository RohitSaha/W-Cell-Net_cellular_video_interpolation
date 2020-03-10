import tensorflow as tf

def huber_loss(prediction, ground_truth,
                delta=1.0):

    # set a small delta value to make it behave like
    # L1 loss
    loss = tf.keras.losses.Huber(
        delta=delta)

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


def ssim_loss(prediction, ground_truth, max_val,
                filter_size=11, filter_sigma=1.5,
                k1=0.01, k2=0.03):

    # Gaussian filter of size 11x11 and width 1.5
    # is used. Image has to be at least 11x11 big.

    ssim_loss = tf.image.ssim(
        prediction,
        ground_truth,
        max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2)

    return ssim_loss
