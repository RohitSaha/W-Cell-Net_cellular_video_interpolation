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


def l1_loss(prediction, ground_truth):
    '''Performs L1 loss between 2 tensors.
    Args:
        :prediction: tensor of shape [N, N_IF, H, W, 1]
        :ground_truth: tensor of shape [N, N_IF, H, W, 1]
    Returns:
        L1 scalar loss of tf.float32
    '''

    # Get absolute loss
    difference = tf.abs(
        prediction - ground_truth)

    # Reduce sum along axis = 1, N_IF
    l1_loss = tf.reduce_sum(
        difference,
        axis=1)

    return tf.reduce_mean(l1_loss)


def l2_loss(prediction, ground_truth):
    '''Performs L2 loss between 2 tensors.
    Args:
        :prediction: tensor of shape [N, N_IF, H, W, 1]
        :ground_truth: tensor of shape [N, N_IF, H, W, 1]
    Returns:
        L2 scalar loss of tf.float32
    '''

    # Get squared loss
    difference = tf.square(
        prediction - ground_truth)

    # Reduce sum along axis=1, N_IF
    # l2_loss = tf.reduce_sum(
    #    difference,
    #    axis=1)

    # return tf.reduce_mean(l2_loss)
    return tf.reduce_mean(difference)


def ridge_weight_decay(parameters):

    weight_decay = tf.add_n(
        [
            tf.nn.l2_loss(param)
            for param in parameters
            if 'w:' in param.name \
            or 'kernel:' in param.name])
    
    return weight_decay


def ssim_loss(prediction, ground_truth, max_val=2.,
                filter_size=11, filter_sigma=1.5,
                k1=0.01, k2=0.03):

    # Try block filter of 8x8
    # Gaussian filter of size 11x11 and width 1.5
    # is used. Image has to be at least 11x11 big.
    # Need to consider scale when integrating with
    # other losses

    ssim_loss = tf.image.ssim(
        prediction,
        ground_truth,
        max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2)

    return 1.0-tf.math.reduce_mean(ssim_loss)

def perceptual_loss(prediction, ground_truth):
    '''Performs L2 loss between 2 tensors.
    Args:
        :prediction: tensor of shape [N, H, W, C]
        :ground_truth: tensor of shape [N, H, W, C]
    Returns:
        L2 scalar loss of tf.float32
    '''

    difference = tf.square(
        prediction - ground_truth)

    # Reduce sum along axis=-1, C
    perceptual_loss = tf.reduce_sum(
        difference,
        axis=-1)

    return tf.reduce_mean(perceptual_loss)
