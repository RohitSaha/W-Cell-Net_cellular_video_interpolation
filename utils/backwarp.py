import tensorflow as tf

def flow_back_wrap(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
    """
      Args:
        x - Input tensor [N, H, W, C]
        v - Vector flow tensor [N, H, W, 2], tf.float32
        (optional)
        resize - Whether to resize v as same size as x
        normalize - Whether to normalize v from scale 1 to H (or W).
                    h : [-1, 1] -> [-H/2, H/2]
                    w : [-1, 1] -> [-W/2, W/2]
        crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
        out  - Handling out of boundary value.
               Zero value is used if out="CONSTANT".
               Boundary values are used if out="EDGE".
    """

    def _get_grid_array(N, H, W, h, w):
        N_i = tf.range(N)
        H_i = tf.range(h + 1, h + H + 1)
        W_i = tf.range(w + 1, w + W + 1)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3)  # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3)  # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3)  # [N, H, W, 1]
        n = tf.cast(n, tf.float32)  # [N, H, W, 1]
        h = tf.cast(h, tf.float32)  # [N, H, W, 1]
        w = tf.cast(w, tf.float32)  # [N, H, W, 1]

        return n, h, w

    shape = tf.shape(x)  # TRY : Dynamic shape
    N = shape[0]
    if crop is None:
        H_ = H = shape[1]
        W_ = W = shape[2]
        h = w = 0
    else:
        H_ = shape[1]
        W_ = shape[2]
        H = crop[1] - crop[0]
        W = crop[3] - crop[2]
        h = crop[0]
        w = crop[2]

    if out == "CONSTANT":
        x = tf.pad(x,((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT')
    elif out == "EDGE":
        x = tf.pad(x,((0, 0), (1, 1), (1, 1), (0, 0)), mode='REFLECT')

    n, h, w = _get_grid_array(N, H, W, h, w)  # [N, H, W, 3] 

    H_1 = tf.cast(H_ + 1, tf.float32)
    W_1 = tf.cast(W_ + 1, tf.float32)  

    original_v = v
    result = None

    for start_idx in range(0,original_v.shape[-1],2):
        v = original_v[...,start_idx:start_idx+2]

        if resize:
            if callable(resize):
                v = resize(v, [H, W])
            else:
                v = tf.image.resize_bilinear(v, [H, W])

        vy, vx = tf.split(v, 2, axis=3)
        if normalize:
            vy = vy * tf.cast(H, dtype=tf.float32)  # TODO: Check why  vy * (H/2) didn't work
            vy = vy / 2
            vx = vy * tf.cast(W, dtype=tf.float32)
            vx = vx / 2

        
        vx0 = tf.floor(vx)
        vy0 = tf.floor(vy)
        vx1 = vx0 + 1
        vy1 = vy0 + 1  # [N, H, W, 1]

        iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
        iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
        ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
        ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

        i00 = tf.concat([n, iy0, ix0], 3)
        i01 = tf.concat([n, iy1, ix0], 3)
        i10 = tf.concat([n, iy0, ix1], 3)
        i11 = tf.concat([n, iy1, ix1], 3)  # [N, H, W, 3]
        i00 = tf.cast(i00, tf.int32)
        i01 = tf.cast(i01, tf.int32)
        i10 = tf.cast(i10, tf.int32)
        i11 = tf.cast(i11, tf.int32)
        x00 = tf.gather_nd(x, i00)
        x01 = tf.gather_nd(x, i01)
        x10 = tf.gather_nd(x, i10)
        x11 = tf.gather_nd(x, i11)
        w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
        w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
        w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
        w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
        output = tf.add_n([w00 * x00, w01 * x01, w10 * x10, w11 * x11])

        if result is None:
            result = output
        else:
            result = tf.concat([result,output],axis=-1)

    return result


# Reference: https://github.com/gunshi/appearance-flow-tensorflow/blob/master/bilinear_sampler.py
def flow_back_wrap_wstack(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
    """
      Args:
        x - Input tensor [N, H, W, C]
        v - Vector flow tensor [N, H, W, 2], tf.float32
        (optional)
        resize - Whether to resize v as same size as x
        normalize - Whether to normalize v from scale 1 to H (or W).
                    h : [-1, 1] -> [-H/2, H/2]
                    w : [-1, 1] -> [-W/2, W/2]
        crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
        out  - Handling out of boundary value.
               Zero value is used if out="CONSTANT".
               Boundary values are used if out="EDGE".
    """

    def _get_grid_array(N, H, W, h, w):
        N_i = tf.range(N)
        H_i = tf.range(h + 1, h + H + 1)
        W_i = tf.range(w + 1, w + W + 1)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3)  # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3)  # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3)  # [N, H, W, 1]
        n = tf.cast(n, tf.float32)  # [N, H, W, 1]
        h = tf.cast(h, tf.float32)  # [N, H, W, 1]
        w = tf.cast(w, tf.float32)  # [N, H, W, 1]

        return n, h, w

    x_shape = x.shape
    iframes = v.shape[-1]//2
    
    # stack along the width
    x = tf.reshape(tf.tile(x,[1,1,1,iframes]),
        [x_shape[0],x_shape[1],-1,x_shape[-1]])


    shape = x.shape  # get new shape of x
   
    N = shape[0]

    if crop is None:
        H_ = H = shape[1]
        W_ = W = shape[2]
        h = w = 0
    else:
        H_ = shape[1]
        W_ = shape[2]
        H = crop[1] - crop[0]
        W = crop[3] - crop[2]
        h = crop[0]
        w = crop[2]

    if resize:
        if callable(resize):
            v = resize(v, [H, W])
        else:
            v = tf.image.resize_bilinear(v, [H, W])

    if out == "CONSTANT":
        x = tf.pad(x,
                   ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT')
    elif out == "EDGE":
        x = tf.pad(x,
                   ((0, 0), (1, 1), (1, 1), (0, 0)), mode='REFLECT')

    vy, vx = tf.split(v, 2, axis=3)
    vy, vx = tf.reshape(vy,shape),tf.reshape(vx,shape)

    if normalize:
        vy = vy * tf.cast(H, dtype=tf.float32)  # TODO: Check why  vy * (H/2) didn't work
        vy = vy / 2
        vx = vy * tf.cast(W, dtype=tf.float32)
        vx = vx / 2

    n, h, w = _get_grid_array(N, H, W, h, w)  # [N, H, W, 3]
    vx0 = tf.floor(vx)
    vy0 = tf.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1  # [N, H, W, 1]

    H_1 = tf.cast(H_ + 1, tf.float32)
    W_1 = tf.cast(W_ + 1, tf.float32)
    iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
    iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
    ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
    ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

    i00 = tf.concat([n, iy0, ix0], 3)
    i01 = tf.concat([n, iy1, ix0], 3)
    i10 = tf.concat([n, iy0, ix1], 3)
    i11 = tf.concat([n, iy1, ix1], 3)  # [N, H, W, 3]
    i00 = tf.cast(i00, tf.int32)
    i01 = tf.cast(i01, tf.int32)
    i10 = tf.cast(i10, tf.int32)
    i11 = tf.cast(i11, tf.int32)
    x00 = tf.gather_nd(x, i00)
    x01 = tf.gather_nd(x, i01)
    x10 = tf.gather_nd(x, i10)
    x11 = tf.gather_nd(x, i11)
    w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
    w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
    w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
    w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
    output = tf.add_n([w00 * x00, w01 * x01, w10 * x10, w11 * x11])

    output = tf.reshape(output,[x_shape[0],x_shape[1],x_shape[2],-1])

    return output



# Reference: https://github.com/gunshi/appearance-flow-tensorflow/blob/master/bilinear_sampler.py
def flow_back_wrap_bstack(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
    """
      Args:
        x - Input tensor [N, H, W, C]
        v - Vector flow tensor [N, H, W, 2], tf.float32
        (optional)
        resize - Whether to resize v as same size as x
        normalize - Whether to normalize v from scale 1 to H (or W).
                    h : [-1, 1] -> [-H/2, H/2]
                    w : [-1, 1] -> [-W/2, W/2]
        crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
        out  - Handling out of boundary value.
               Zero value is used if out="CONSTANT".
               Boundary values are used if out="EDGE".
    """

    def _get_grid_array(N, H, W, h, w):
        N_i = tf.range(N)
        H_i = tf.range(h + 1, h + H + 1)
        W_i = tf.range(w + 1, w + W + 1)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3)  # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3)  # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3)  # [N, H, W, 1]
        n = tf.cast(n, tf.float32)  # [N, H, W, 1]
        h = tf.cast(h, tf.float32)  # [N, H, W, 1]
        w = tf.cast(w, tf.float32)  # [N, H, W, 1]

        return n, h, w

    x_shape = x.shape
    iframes = v.shape[-1]//2

    x = tf.tile(x,[iframes,1,1,1]) #stack along batch

    shape = x.shape  # get new shape of x
   
    N = shape[0]

    if crop is None:
        H_ = H = shape[1]
        W_ = W = shape[2]
        h = w = 0
    else:
        H_ = shape[1]
        W_ = shape[2]
        H = crop[1] - crop[0]
        W = crop[3] - crop[2]
        h = crop[0]
        w = crop[2]

    if resize:
        if callable(resize):
            v = resize(v, [H, W])
        else:
            v = tf.image.resize_bilinear(v, [H, W])

    if out == "CONSTANT":
        x = tf.pad(x,
                   ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT')
    elif out == "EDGE":
        x = tf.pad(x,
                   ((0, 0), (1, 1), (1, 1), (0, 0)), mode='REFLECT')

    vy, vx = tf.split(v, 2, axis=3)
    vy, vx = tf.reshape(vy,shape),tf.reshape(vx,shape)

    if normalize:
        vy = vy * tf.cast(H, dtype=tf.float32)  # TODO: Check why  vy * (H/2) didn't work
        vy = vy / 2
        vx = vy * tf.cast(W, dtype=tf.float32)
        vx = vx / 2

    n, h, w = _get_grid_array(N, H, W, h, w)  # [N, H, W, 3]
    vx0 = tf.floor(vx)
    vy0 = tf.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1  # [N, H, W, 1]

    H_1 = tf.cast(H_ + 1, tf.float32)
    W_1 = tf.cast(W_ + 1, tf.float32)
    iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
    iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
    ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
    ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

    i00 = tf.concat([n, iy0, ix0], 3)
    i01 = tf.concat([n, iy1, ix0], 3)
    i10 = tf.concat([n, iy0, ix1], 3)
    i11 = tf.concat([n, iy1, ix1], 3)  # [N, H, W, 3]
    i00 = tf.cast(i00, tf.int32)
    i01 = tf.cast(i01, tf.int32)
    i10 = tf.cast(i10, tf.int32)
    i11 = tf.cast(i11, tf.int32)
    x00 = tf.gather_nd(x, i00)
    x01 = tf.gather_nd(x, i01)
    x10 = tf.gather_nd(x, i10)
    x11 = tf.gather_nd(x, i11)
    w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
    w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
    w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
    w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
    output = tf.add_n([w00 * x00, w01 * x01, w10 * x10, w11 * x11])

    output = tf.reshape(output,[x_shape[0],x_shape[1],x_shape[2],-1])

    return output