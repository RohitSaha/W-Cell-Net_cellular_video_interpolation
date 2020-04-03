import tensorflow as tf

from models.utils.layers import linear as MLP
from models.utils.layers import conv_batchnorm_relu as CBR
from models.utils.layers import upconv_2D as UC
from models.utils.layers import maxpool as MxP

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
            conv_1, 'conv_2', out_channels,
            activation=tf.keras.activations.relu,
            kernel_size=kernel_size, stride=stride,
            is_training=is_training,
            use_batch_norm=use_batch_norm)

    return conv_2


def discriminator(inputs, use_batch_norm=False,
                is_training=False, is_verbose=False,
                starting_out_channels=8):

    layer_dict = {}

    get_shape = inputs.get_shape().as_list()
    if is_verbose: print('Inputs:{}'.format(get_shape))

    # [N, 100, 100, 1]
    block_1 = conv_block(
        inputs, block_name='block_1',
        out_channels=starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    block_1 = MxP(
        block_1,
        'MxP_1',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Block_1:{}'.format(block_1))
    layer_dict['block_1'] = block_1

    # [N, 50, 50, 8]
    block_2 = conv_block(
        block_1, block_name='block_2',
        out_channels=2*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    block_2 = MxP(
        block_2,
        'MxP_2',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Block_2:{}'.format(block_2))
    layer_dict['block_2'] = block_2

    # [N, 25, 25, 16]
    block_3 = conv_block(
        block_2, block_name='block_3',
        out_channels=4*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    block_3 = MxP(
        block_3,
        'MxP_3',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='VALID')
    if is_verbose: print('Block_3:{}'.format(block_3))
    layer_dict['block_3'] = block_3

    # [N, 12, 12, 32]
    block_4 = conv_block(
        block_3, block_name='block_4',
        out_channels=8*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    block_4 = MxP(
        block_4,
        'MxP_4',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Block_4:{}'.format(block_4))
    layer_dict['block_4'] = block_4

    # [N, 6, 6, 64]
    block_5 = conv_block(
        block_4, block_name='block_5',
        out_channels=16*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    block_5 = MxP(
        block_5,
        'MxP_5',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Block_5:{}'.format(block_5))
    layer_dict['block_5'] = block_5

    # [N, 3, 3, 128]
    block_6 = conv_block(
        block_5, block_name='block_5',
        out_channels=16*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    block_6 = MxP(
        block_6,
        'MxP_6',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Block_6:{}'.format(block_6))
    layer_dict['block_6'] = block_6

    return block_6, layer_dict


def fully_connected_layer(inputs, is_training=False,
                        is_verbose=False):

    N, H, W, C = inputs.get_shape().as_list()
    inputs = tf.reshape(
        inputs,
        [N, -1])

    # [N, 1152]
    net = MLP(
        inputs,
        'MLP_1',
        1,
        activation=None,
        batch_norm=False,
        is_training=is_training)

    if is_verbose: print('MLP_1:{}'.format(net))
    # [N, 1]

    return net

         
def build_discriminator(inputs, use_batch_norm=False,
                        is_training=False,
                        starting_out_channels=8,
                        is_verbose=False):

    N, N_IF, H, W, C = inputs.get_shape().as_list()
    # [N, N_IF, H, W, 1] -> [N, N_IF, H, W]
    inputs = tf.reshape(
        inputs,
        [N, N_IF, H, W])
    # [N, N_IF, H, W] -> [N, H, W, N_IF]
    inputs = tf.transpose(
        inputs,
        [0, 2, 3, 1])

    print('Discriminator......')
    with tf.variable_scope('discriminator'):
        features, layer_dict = discriminator(
            inputs,
            use_batch_norm=use_batch_norm,
            is_training=is_training,
            is_verbose=is_verbose,
            starting_out_channels=starting_out_channels)

        # [N, 3, 3, 128]
        features = fully_connected_layer(
            inputs,
            is_training=is_training,
            is_verbose=is_verbose)

    return features 
