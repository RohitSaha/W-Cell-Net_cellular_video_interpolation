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


def ssim_loss(prediction, ground_truth, max_val=2.,
                filter_size=11, filter_sigma=1.5,
                k1=0.01, k2=0.03):

    # Try block filter of 8x8
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



def perceptual_loss(mid_frames, rec_mid_frames, model, 
    sess=tf.Session(),layer_name = 'conv4_3', verbose=False):

    shape = mid_frames.get_shape().as_list()

    try:
        layer = getattr(model,layer_name)
    except:
        print('Layer not available in model')

    img_min = tf.math.reduce_min(mid_frames)
    img_max = tf.math.reduce_max(mid_frames)
    fk_img_min = tf.math.reduce_min(rec_mid_frames) 
    fk_img_max = tf.math.reduce_max(rec_mid_frames)

    img_range = img_max - img_min
    fk_img_range = fk_img_max - fk_img_min

    mid_frames = 255 * \
        (mid_frames - img_min) / img_range
    rec_mid_frames = 255 * \
        (rec_mid_frames - fk_img_min)/ fk_img_range 
    

    if len(shape)>4:
        mid_frames = tf.reshape(mid_frames,\
            [-1,shape[-3],shape[-2],shape[-1]])
        rec_mid_frames = tf.reshape(rec_mid_frames,\
            [-1,shape[-3],shape[-2],shape[-1]])
    elif len(shape)==3:
        mid_frames = tf.expand_dims(mid_frames,0)
        rec_mid_frames = tf.expand_dims(\
            rec_mid_frames,0)

    mid_scaled = tf.image.resize(mid_frames,\
        (224,224))
    rec_mid_scaled = tf.image.resize(rec_mid_frames,\
        (224,224))

    
    mid_scaled_rgb = tf.image.grayscale_to_rgb(\
        mid_scaled)
    rec_mid_scaled_rgb = tf.image.grayscale_to_rgb(\
        rec_mid_scaled)
    
    mid_img, rec_mid_img = sess.run([mid_scaled_rgb,\
        rec_mid_scaled_rgb])


    mid_img_features = sess.run(layer,\
        feed_dict={model.imgs: mid_img})
    rec_mid_img_features = sess.run(layer,\
        feed_dict={model.imgs: rec_mid_img})


    percep_error = tf.nn.l2_loss(\
        rec_mid_img_features-mid_img_features)

    if verbose:
        print('Error calculated for layers {}'\
            .format(layer.name))

    return percep_error
