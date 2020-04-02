import tensorflow as tf

from models.utils.layer import linear as MLP
from models.utils.layer import conv_batchnorm_relu as CBR
from models.utils.layer import upconv_2D as UC
from models.utils.layer import maxpool as MxP
from models.utils.layer import avgpool as AvP
from models.utils.layer import spatial_attention as SAttn
from models.utils.layer import channel_attention as CAttn

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
 

def encoder(inputs, use_batch_norm=False,
            is_training=False, is_verbose=False,
            starting_out_channels=8):

    layer_dict = {}

    get_shape = inputs.get_shape().as_list()
    if is_verbose: print('Inputs:{}'.format(get_shape))

    encode_1 = conv_block(
        inputs, block_name='block_1',
        out_channels=starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    encode_1 = MxP(
        encode_1,
        'MxP_1',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Encode_1:{}'.format(encode_1))
    layer_dict['encode_1'] = encode_1

    encode_2 = conv_block(
        encode_1, block_name='block_2',
        out_channels=2*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    encode_2 = MxP(
        encode_2,
        'MxP_2',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Encode_2:{}'.format(encode_2))
    layer_dict['encode_2'] = encode_2

    encode_3 = conv_block(
        encode_2, block_name='block_3',
        out_channels=4*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    encode_3 = MxP(
        encode_3,
        'MxP_3',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='VALID')
    if is_verbose: print('Encode_3:{}'.format(encode_3))
    layer_dict['encode_3'] = encode_3

    encode_4 = conv_block(
        encode_3, block_name='block_4',
        out_channels=8*starting_out_channels,
        kernel_size=3,
        stride=1,
        use_batch_norm=use_batch_norm,
        is_training=is_training)
    encode_4 = MxP(
        encode_4,
        'MxP_4',
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        padding='SAME')
    if is_verbose: print('Encode_4:{}'.format(encode_4))
    layer_dict['encode_4'] = encode_4

    return encode_4, layer_dict

def upconv_block(inputs, block_name='block_1',
                use_batch_norm=False,
                kernel_size=3, stride=1, use_bias=False,
                out_channels=16, is_training=False):

    # upconv(x2, c/2) --> 2 convs
    
    with tf.variable_scope(block_name):
        net = UC(inputs, 'up_conv', out_channels,
            kernel_size=(2, 2), strides=(2, 2),
            use_bias=use_bias)

        if block_name == 'block_2':
            # BILINEAR RESIZE
            net = tf.image.resize_images(
                net, (25, 25),
                align_corners=True)

        # Use tanh for the last decoder conv layer
        if block_name == 'block_4':
            activation = tf.keras.activations.tanh
        else:
            activation = tf.keras.activations.relu

        activation = tf.keras.activations.tanh

        for i in range(1): 
            net = CBR(
                net, 'conv_{}'.format(str(i)), out_channels,
                activation=activation, # tanh
                kernel_size=kernel_size, stride=stride,
                is_training=is_training,
                use_batch_norm=use_batch_norm)

    return net


def decoder(inputs, layer_dict_fFrames,
            layer_dict_lFrames, use_batch_norm=False,
            n_IF=3, is_training=False,
            is_verbose=False, use_attention=0,
            spatial_attention=0):

    get_shape = inputs.get_shape().as_list()
    out_channels = get_shape[-1]

    decode_1 = upconv_block(
        inputs,
        block_name='block_1',
        use_batch_norm=True,
        kernel_size=3, stride=1,
        out_channels=out_channels//2,
        use_bias=True)
    if is_verbose: print('Decode_1:{}'.format(decode_1))

    # add skip connection
    fFrames_encode_3 = layer_dict_fFrames['encode_3']
    lFrames_encode_3 = layer_dict_lFrames['encode_3']
    decode_1 = tf.concat(
        [
            fFrames_encode_3,
            decode_1,
            tf.reverse(
                lFrames_encode_3,
                axis=[-1])],
        axis=-1)

    if use_attention:
        if spatial_attention:
            decode_1 = SAttn(decode_1)
            if is_verbose: print('SpatialAttn_1:{}'.format(decode_1))
        else:
            decode_1 = CAttn(decode_1)
            if is_verbose: print('ChannelAttn_1:{}'.format(decode_1))

    if is_verbose: print('MergeDecode_1:{}'.format(decode_1))
    # decode_1 channels: 256
    
    get_shape = decode_1.get_shape().as_list()
    out_channels = get_shape[-1]

    decode_2 = upconv_block(
        decode_1,
        block_name='block_2',
        use_batch_norm=True,
        kernel_size=3, stride=1,
        out_channels=out_channels//2,
        use_bias=True)
    if is_verbose: print('Decode_2:{}'.format(decode_2))

    # add skip connection
    fFrames_encode_2 = layer_dict_fFrames['encode_2']
    lFrames_encode_2 = layer_dict_fFrames['encode_2']
    decode_2 = tf.concat(
        [
            fFrames_encode_2,
            decode_2,
            tf.reverse(
                lFrames_encode_2,
                axis=[-1])],
        axis=-1)

    if use_attention:
        if spatial_attention:
            decode_2 = SAttn(decode_2)
            if is_verbose: print('SpatialAttn_2:{}'.format(decode_2))
        else:
            decode_2 = CAttn(decode_2)
            if is_verbose: print('ChannelAttn_2:{}'.format(decode_2))

    if is_verbose: print('MergeDecode_2:{}'.format(decode_2))
    # decode_2 channels: 192

    decode_3 = upconv_block(
        decode_2,
        block_name='block_3',
        use_batch_norm=True,
        kernel_size=3, stride=1,
        out_channels=64,
        use_bias=True)
    if is_verbose: print('Decode_3:{}'.format(decode_3))

    # add skip connection
    fFrames_encode_1 = layer_dict_fFrames['encode_1']
    lFrames_encode_1 = layer_dict_lFrames['encode_1']
    decode_3 = tf.concat(
        [
            fFrames_encode_1,
            decode_3,
            tf.reverse(
                lFrames_encode_1,
                axis=[-1])],
        axis=-1)

    if use_attention:
        if spatial_attention:
            decode_3 = SAttn(decode_3)
            if is_verbose: print('SpatialAttn_3:{}'.format(decode_3))
        else:
            decode_3 = CAttn(decode_3)
            if is_verbose: print('ChannelAttn_3:{}'.format(decode_3))

    if is_verbose: print('MergeDecode_3:{}'.format(decode_3))
    # decode_3 channels: 96

    decode_4 = upconv_block(
        decode_3,
        block_name='block_4',
        use_batch_norm=True,
        kernel_size=3, stride=1,
        out_channels=n_IF,
        use_bias=True)
    if is_verbose: print('Decode_4:{}'.format(decode_4))
               
    return decode_4


def build_bipn(fFrames, lFrames, n_IF=3, use_batch_norm=False,
                is_training=False, starting_out_channels=8,
                use_attention=0, input_layer_skip=False,
                spatial_attention=0):

    with tf.variable_scope('encoder_1'):
        encode_fFrames, layer_dict_fFrames = encoder(
            fFrames,
            use_batch_norm=use_batch_norm,
            is_training=is_training,
            is_verbose=True,
            starting_out_channels=starting_out_channels)

    # use same encoder weights for last frame
    with tf.variable_scope('encoder_2'):
        encode_lFrames, layer_dict_lFrames = encoder(
            lFrames,
            use_batch_norm=use_batch_norm,
            is_training=is_training,
            is_verbose=False,
            starting_out_channels=starting_out_channels)

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

    with tf.variable_scope('decoder'):
        rec_iFrames = decoder(
            encode_Frames,
            layer_dict_fFrames,
            layer_dict_lFrames,
            n_IF=n_IF,
            use_batch_norm=use_batch_norm,
            is_training=is_training,
            is_verbose=True,
            use_attention=use_attention,
            spatial_attention=spatial_attention)

    if input_layer_skip:
        # adding skip connection at the input layer
        rec_iFrames = tf.concat(
            [fFrames, rec_iFrames, lFrames],
            axis=-1)
        get_shape = rec_iFrames.get_shape().as_list()
        print('Skip connection at input layer:{}'.format(
            get_shape))

        '''
        # adding residual connection
        rec_iFrames = rec_iFrames + fFrames
        print('Residual connection:{}'.format(
            rec_iFrames.get_shape().as_list()))
        '''

        # 3x3 conv layer to reduce channels to :
        with tf.variable_scope('final_conv'):
            rec_iFrames = CBR(
                rec_iFrames, 'conv_final', n_IF,
                activation=tf.keras.activations.relu,
                kernel_size=3, stride=1,
                is_training=is_training,
                use_batch_norm=use_batch_norm)

    rec_iFrames = tf.transpose(
        rec_iFrames,
        [0, 3, 1, 2])
    rec_iFrames = tf.expand_dims(
        rec_iFrames,
        axis=-1)
    print('Final decoder:{}'.format(rec_iFrames))

    return rec_iFrames

