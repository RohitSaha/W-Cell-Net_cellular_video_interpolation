import tensorflow as tf

from models.utils.layer import linear as MLP
from models.utils.layer import conv_batchnorm_relu as CBR
from models.utils.layer import upconv_2D as UC
from models.utils.layer import maxpool as MxP
from models.utils.layer import avgpool as AvP

def conv_block(inputs, block_name='block_1',
                out_channels=16,
                kernel_size=3,
                stride=1,
                use_batch_norm=False,
                is_training=False):

    get_shape = inputs.get_shape().as_list()

    with tf.variable_scope(block_name):
        conv_1 = CBR(
            inputs, 'conv_1', out_channels,
            activation=tf.keras.activations.relu,
            kernel_size=kernel_size, stride=stride,
            is_training=is_training,
            use_batch_norm=use_batch_norm)

        conv_2 = CBR(
            conv_1, 'conv_2', out_channels*2,
            activation=tf.keras.activations.relu,
            kernel_size=kernel_size, stride=stride,
            is_training=is_training,
            use_batch_norm=use_batch_norm)

    return conv_2
 

def encoder(inputs, use_batch_norm=False,
            is_training=False):

    get_shape = inputs.get_shape().as_list()
    print('Inputs:{}'.format(get_shape))

    net = conv_block(
        inputs, block_name='block_1',
        out_channels=16, kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    net = MxP(
        net,
        'MxP_1',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    print('Encode_1:{}'.format(net))

    net = conv_block(
        net, block_name='block_2',
        out_channels=32, kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    net = MxP(
        net,
        'MxP_2',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    print('Encode_2:{}'.format(net))

    net = conv_block(
        net, block_name='block_3',
        out_channels=64, kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    net = MxP(
        net,
        'MxP_3',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    print('Encode_3:{}'.format(net))

    net = conv_block(
        net, block_name='block_4',
        out_channels=128, kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    net = MxP(
        net,
        'MxP_4',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    print('Encode_4:{}',format(net))

    return net


def decoder(inputs, use_batch_norm=False,
            is_training=False):

    get_shape = inputs.get_shape().as_list()
    return inputs


def build_bipn(fFrames, lFrames, use_batch_norm=False,
                is_training=False):

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

    # Flip :encode_lFrames
    # not too confident about tf.reverse behavior
    encode_lFrames = tf.reverse(
        encode_lFrames,
        axis=[-1])

    # Concatenate :encode_fFrames and :encode_lFrames
    encode_Frames = tf.concat(
        [encode_fFrames, encode_lFrames],
        axis=-1)
    print('Concatenated:{}'.format(
        encode_Frames.get_shape().as_list()))

    return encode_Frames

    with tf.variable_scope('decoder'):
        rec_iFrames = decoder(
            encode_Frames,
            use_batch_norm=use_batch_norm,
            is_training=is_training)

    return rec_iFrames 
