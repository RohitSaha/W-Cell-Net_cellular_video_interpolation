import tensorflow as tf
import math as pymath

def gaussian_kernel (k_size=(7,7),mean=0,std=1):
    '''
    creates 2 probability ditributions 
    (p_dist_1,p_dist_2) with lengths determined 
    by k_size and the does the outer product

    Args:
        k_size: tuple, [kernel_height, kernel_width]
        mean: scalar, gaussian distribution mean
        std: scalar, gaussian distribution deviation

    Output:
        gauss_kernel: [kernel_height, kernel_width]
                    gaussian weights
    '''

    p_dist_1 = tf.map_fn(lambda x : 
        tf.math.exp(-0.5*((x-mean)/std)**2)/(
            std*tf.math.sqrt(2*pymath.pi)),
        tf.range((-k_size[0]+1)//2,(k_size[0]+1)//2,
            dtype=tf.float32))

    p_dist_2 = tf.map_fn(lambda x : 
        tf.math.exp(-0.5*((x-mean)/std)**2)/(
            std*tf.math.sqrt(2*m.pi)),
        tf.range((-k_size[1]+1)//2,(k_size[1]+1)//2,
            dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',p_dist_1, 
        p_dist_2)

    return gauss_kernel/tf.math.reduce_sum(
        gauss_kernel)

def gaussian_filter(img,k_size=(7,7),mean=0,std=3):
    
    '''
    Create a gaussian kernel of shape k_size and
    convolve with image to generate filtered image

    Args:
        img: tensors [B,H,W,1] or [B,#frames,H,W,1]
        k_size: tuple, [kernel_height, kernel_width]
        mean: scalar, gaussian distribution mean
        std: scalar, gaussian distribution deviation

    Output:
        blurred_image: same shape as img
    '''
    img_shape = img.get_shape()

    if len(img_shape)==5:
        
        img = tf.reshape(tf.transpose(
        img,perm=[0,2,1,3,4]),
        [img_shape[0],img_shape[2],
        -1,img_shape[4]])


    width = k_size[0]
    height = k_size[1]

    g_kernel = gaussian_kernel(k_size=k_size,
        mean=mean,std=std)

    blurred_img = tf.nn.conv2d(img,
        tf.reshape(g_kernel,[width,height,1,1]),
        strides=[1,1,1,1],padding='SAME')

    blurred_img = tf.stop_gradient(blurred_img)

    if len(img_shape)==5:

        blurred_img = tf.transpose(tf.reshape(
        blurred_img,[img_shape[0],img_shape[2],
        -1,img_shape[3],img_shape[4]]),
        perm=[0,2,1,3,4])

    return blurred_img

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

