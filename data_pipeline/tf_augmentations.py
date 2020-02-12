import tensorflow as tf

def random_brightness(frames):

    delta_var = tf.random_uniform(
        shape=[],
        dtype=tf.float32) * 0.2

    frames = tf.image.adjust_brightness(
        frames,
        delta=delta_var)

    return frames


def random_contrast(frames):
    
    frames = tf.image.random_contrast(
        frames,
        0.9,
        1.1)

    return frames

def random_lr_flip(frames):

    def __lr_flip(frames):
        frames = tf.image.flip_left_right(
            frames)

        return frames

    def no_flip(frames):
        return frames

    frames = tf.cond(
        tf.random_uniform(
            shape=[],
            dtype=tf.float32) > 0.5,
        lambda: __lr_flip(frames),
        lambda: no_flip(frames))

    return frames


def random_ud_flip(frames):

    def __ud_flip(frames):
        frames = tf.image.flip_up_down(
            frames)

        return frames

    def no_flip(frames):
        return frames

    frames = tf.cond(
        tf.random_uniform(
            shape=[],
            dtype=tf.float32) > 0.5,
        lambda: __ud_flip(frames),
        lambda: no_flip(frames))

    return frames

def augment(fFrame, lFrame, iFrame):

    # Transpose and slice to get [H, W, C]
    iFrame = tf.transpose(
        iFrame,
        [3, 1, 2, 0])[0, ...]

    frames = tf.concat(
        [fFrame, iFrame, lFrame],
        axis=-1)

    frames = random_contrast(
        random_brightness(
            random_lr_flip(
                random_ud_flip(
                    frames))))

    
    # Slice :frames to get back individual frames
    fFrame = tf.expand_dims(
        frames[..., 0],
        axis=-1)

    lFrame = tf.expand_dims(
        frames[..., -1],
        axis=-1)

    iFrame = tf.expand_dims(
        frames[..., 1:-1],
        axis=0)
    iFrame = tf.transpose(
        iFrame,
        [3, 1, 2, 0])

    return fFrame, lFrame, iFrame 

