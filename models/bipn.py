import tensorflow as tf

from models.utils.layer import linear as MLP
from models.utils.layer import conv_batchnorm_relu as CBR
from models.utils.layer import upconv_2D as UC
from models.utils.layer import maxpool as MxP
from models.utils.layer import avgpool as AvP


def encoder(inputs, use_batch_norm=False,
            is_training=False):

    get_shape = inputs.get_shape().as_list()
    return inputs

def decoder(inputs, use_batch_norm=False,
            is_training=False):

    get_shape = inputs.get_shape().as_list()
    return inputs


def build_bipn(fFrames, lFrames, use_batch_norm=False,
                is_training=False):

    get_shape = inputs.get_shape().as_list()

    with tf.variable_scope('encoder'):
        encode_fFrames = encoder(
            fFrames,
            use_batch_norm=use_batch_norm,
            is_training=is_training)

    # use same encoder weights for last frame
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        encode_lFrames = encoder(
            lFrames,
            use_batch_norm=use_batch_norm,
            is_training=is_training)

    # TODO: Flip :encode_lFrames


    # Concatenate :encode_fFrames and :encode_lFrames
    encode_Frames = tf.concat(
        [encode_fFrames, encode_lFrames],
        axis=-1)

    with tf.variable_scope('decoder'):
        rec_iFrames = decoder(
            encode_Frames,
            use_batch_norm=use_batch_norm,
            is_training=is_training)

    return rec_iFrames 
