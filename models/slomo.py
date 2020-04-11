from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.losses import l1_loss
from utils.backwarp import flow_back_wrap_wstack as flow_back_wrap



def l1_loss(Ipred, Iref, axis=[3]):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(Ipred - Iref), axis=axis))  # L1 Norm


def l2_loss(Ipred, Iref, axis=[3]):
    return tf.reduce_mean(tf.reduce_sum(tf.square(Ipred - Iref), axis=axis))  # L2 Norm


def wrapping_loss(frame0, frame1, frameT, F01, F10, Fdasht0, Fdasht1):
    frameT = tf.squeeze(tf.transpose(frameT,[0,2,3,1,4]),axis=-1)
    return l1_loss(frame0, flow_back_wrap(frame1, F01)) + \
           l1_loss(frame1, flow_back_wrap(frame0, F10)) + \
           l1_loss(frameT, flow_back_wrap(frame0, Fdasht0)) + \
           l1_loss(frameT, flow_back_wrap(frame1, Fdasht1))


def smoothness_loss(F01, F10):
    deltaF01 = tf.reduce_mean(tf.abs(F01[:, 1:, :, :] - F01[:, :-1, :, :])) + tf.reduce_mean(
        tf.abs(F01[:, :, 1:, :] - F01[:, :, :-1, :]))
    deltaF10 = tf.reduce_mean(tf.abs(F10[:, 1:, :, :] - F10[:, :-1, :, :])) + tf.reduce_mean(
        tf.abs(F10[:, :, 1:, :] - F10[:, :, :-1, :]))
    return 0.5 * (deltaF01 + deltaF10)


# Model Helper Functions
def conv2d(batch_input, output_channels, kernel_size=3, 
    stride=1, scope="conv", activation=None):
    with tf.variable_scope(scope):
        activation_fn = None
        if activation == 'leaky_relu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        elif activation == 'relu':
            activation_fn = tf.nn.relu

        return slim.conv2d(batch_input, output_channels,
            [kernel_size, kernel_size], stride=stride,
            data_format='NHWC',
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=activation_fn)


def lrelu(input, alpha=0.2):
    return tf.nn.leaky_relu(input, alpha=alpha)


def average_pool(input, kernel_size, stride=2, scope="avg_pool"):
    return tf.contrib.layers.avg_pool2d(input,
        [kernel_size, kernel_size], stride, scope=scope)


def bilinear_upsampling(input, scale=2, scope="bi_upsample"):
    with tf.variable_scope(scope):
        shape = tf.shape(input)
        h, w = shape[1], shape[2]
        return tf.image.resize_bilinear(input, [scale * h, scale * w])


def encoder_block(inputs,output_channel,conv_kernel=3,
    pool_kernel=2,lrelu_alpha=0.1,scope="enc_block"):
    
    with tf.variable_scope(scope):
        net = conv2d(inputs, output_channel, kernel_size=conv_kernel)
        conv = lrelu(net, lrelu_alpha)
        pool = average_pool(conv, pool_kernel)
        return conv, pool


def decoder_block(input,skip_conn_input,output_channel,
    conv_kernel=3,up_scale=2,lrelu_alpha=0.1,
    scope="dec_block"):

    with tf.variable_scope(scope):
        upsample = bilinear_upsampling(input, scale=up_scale)

        upsample_shape = tf.shape(upsample)  # get_shape() - Static, Tf.shape() = dynamic
        skip_conn_shape = tf.shape(skip_conn_input)

        # upsample shape can differ from skip conn input (becouse of avg-pool and then bi-upsample in case of odd shape)
        xdiff, ydiff = skip_conn_shape[1] - upsample_shape[1], skip_conn_shape[2] - upsample_shape[2]
        upsample = tf.pad(upsample, tf.convert_to_tensor(\
            [[0, 0], [0, xdiff], [0, ydiff], [0, 0]], dtype=tf.int32))
        block_input = tf.concat([upsample, skip_conn_input], 3)

        net = conv2d(block_input, output_channel, kernel_size=conv_kernel)
        net = lrelu(net, lrelu_alpha)
        return net


def UNet(inputs, output_channels, decoder_extra_input=None,
    first_kernel=7, second_kernel=5, scope='unet',
    output_activation=None, reuse=False):
    
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("encoder"):
            econv1, epool1 = encoder_block(inputs, 32, conv_kernel=first_kernel, scope="en_conv1")
            econv2, epool2 = encoder_block(epool1, 64, conv_kernel=second_kernel, scope="en_conv2")
            econv3, epool3 = encoder_block(epool2, 128, scope="en_conv3")
            econv4, epool4 = encoder_block(epool3, 256, scope="en_conv4")
            econv5, epool5 = encoder_block(epool4, 512, scope="en_conv5")
            with tf.variable_scope("en_conv6"):
                econv6 = conv2d(epool5, 512)
                econv6 = lrelu(econv6, alpha=0.1)

        with tf.variable_scope("decoder"):
            decoder_input = econv6
            if decoder_extra_input is not None:
                decoder_input = tf.concat([decoder_input, decoder_extra_input], axis=3)
            net = decoder_block(decoder_input, econv5, 512, scope="dec_conv1")
            net = decoder_block(net, econv4, 256, scope="dec_conv2")
            net = decoder_block(net, econv3, 128, scope="dec_conv3")
            net = decoder_block(net, econv2, 64, scope="dec_conv4")
            net = decoder_block(net, econv1, 32, scope="dec_conv5")

        with tf.variable_scope("unet_output"):
            net = conv2d(net, output_channels, scope="output")
            if output_activation is not None:
                if output_activation == "tanh":
                    net = tf.nn.tanh(net)
                elif output_activation == "lrelu":
                    net = lrelu(net, alpha=0.1)
                else:
                    raise ValueError("only lrelu|tanh allowed")
            return net, econv6


# SloMo vanila model
def SloMo_model(frame0,frame1,first_kernel=7,
    second_kernel=5,reuse=False,t_steps=1,
    verbose=False):
    
    epsilon = 1e-12
   
    if t_steps>1:
        timestamp = tf.range(1.0/(t_steps+1),1,
            delta=1.0/(t_steps+1),dtype=tf.float32)
    else:
        timestamp = 0.5

    shape = frame0.get_shape()
    with tf.variable_scope("SloMo_model", reuse=reuse):
        with tf.variable_scope("flow_computation"):
            flow_comp_input = tf.concat([frame0, frame1], axis=3)
            flow_comp_out, flow_comp_enc_out = UNet(flow_comp_input,
                                                    output_channels=4,  # 2 channel for each flow
                                                    first_kernel=first_kernel,
                                                    second_kernel=second_kernel)
            flow_comp_out = lrelu(flow_comp_out)
            F01, F10 = flow_comp_out[:, :, :, :2], flow_comp_out[:, :, :, 2:]

            if verbose:
                print("Flow Computation Graph Initialized !!!!!! ")

            if t_steps>1:
                timestamp = tf.range(1.0/(t_steps+1),1,
                    delta=1.0/(t_steps+1),dtype=tf.float32)
                F01_temp, F10_temp = tf.expand_dims(F01,axis=-1),\
                                    tf.expand_dims(F10,axis=-1)
            else:
                timestamp = 0.5
                F01_temp, F10_temp = F01, F10

        with tf.variable_scope("flow_interpolation"):
            Fdasht0 = tf.reshape((-1 * (1 - timestamp) * timestamp * F01_temp) + 
                (timestamp * timestamp * F10_temp),[shape[0],shape[1],shape[2],-1])
            Fdasht1 = tf.reshape(((1 - timestamp) * (1 - timestamp) * F01_temp) - 
                (timestamp * (1 - timestamp) * F10_temp),[shape[0],shape[1],shape[2],-1])


            flow_interp_input = tf.concat([frame0, frame1,
                                           flow_back_wrap(frame1, Fdasht1),
                                           flow_back_wrap(frame0, Fdasht0),
                                           Fdasht0, Fdasht1], axis=3)
            # out_channels = 2 channels for each flow and each time_step + time_step visibilty maps.
            flow_interp_output, _ = UNet(flow_interp_input,
                                         output_channels=(4*t_steps+t_steps),  
                                         decoder_extra_input=flow_comp_enc_out,
                                         first_kernel=3,
                                         second_kernel=3)


            deltaFt0, deltaFt1, Vt0 = flow_interp_output[:, :, :, :2*t_steps],\
                                      flow_interp_output[:, :, :, 2*t_steps:4*t_steps],\
                                      flow_interp_output[:, :, :, 4*t_steps:]


            deltaFt0 = lrelu(deltaFt0)
            deltaFt1 = lrelu(deltaFt1)
            Vt0 = tf.sigmoid(Vt0)
            Vt1 = 1 - Vt0

            Ft0, Ft1 = Fdasht0 + deltaFt0, Fdasht1 + deltaFt1

            normalization_factor = 1 / ((1 - timestamp) * Vt0 + timestamp * Vt1 + epsilon)
            pred_frameT = tf.multiply((1 - timestamp) * Vt0, flow_back_wrap(frame0, Ft0)) + \
                          tf.multiply(timestamp * Vt1, flow_back_wrap(frame1, Ft1))
            pred_frameT = tf.multiply(normalization_factor, pred_frameT)
            
            if verbose:
                print("Flow Interpolation Graph Initialized !!!!!! ")

            prediction = tf.expand_dims(tf.transpose(pred_frameT,[0,3,1,2]),axis=-1)

            return prediction,F01, F10, Fdasht0, Fdasht1


# import time

# start = time.time()
# x = tf.random.uniform(
#     shape = [32,100,100,1], minval=0, maxval=1, \
#     dtype=tf.dtypes.float32, seed=None, name=None)

# y = tf.random.uniform(
#     shape = [32,100,100,1], minval=0, maxval=1, \
#     dtype=tf.dtypes.float32, seed=None, name=None)


# avg_f = (x+y)/2

# pred,flow_01,flow_10,weighted_ft0,weightedft1 = SloMo_model(x,y,t_steps=2)

# with tf.Session().as_default() as sess:
#     init_op = tf.group(
#             tf.global_variables_initializer(),
#             tf.local_variables_initializer())
#     saver = tf.train.Saver()

#     sess.run(init_op)
#     pred,flow_01,flow_10,weighted_ft0,weightedft1 = sess.run([pred,flow_01,flow_10,weighted_ft0,weightedft1] )
#     w_loss = sess.run(wrapping_loss(x,y,pred,flow_01,flow_10,weighted_ft0,weightedft1))
#     print(w_loss)

# end = time.time()

# print('time elapsed',end-start)

# pred_out = SloMo_model(x,y,t_steps=5)
# print(pred_out[0].shape)
# print(pred_out[1].shape)
# print(pred_out[2].shape)
# print(pred_out[3].shape)
# print(pred_out[4].shape)
# print(flow_01.shape,flow_10.shape,weighted_ft0.shape,weightedft1.shape)
