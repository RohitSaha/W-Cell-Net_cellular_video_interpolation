import os

import tensorflow as tf
from data_pipeline import tf_augmentations

def read_and_decode(filename_queue=[], is_training=False,
                    batch_size=32, height=100, width=100,
                    n_intermediate_frames=3):

    reader = tf.TFRecordReader()
    _, ser = reader.read(
        filename_queue)

    keys_to_features = {
        'data/first_frame': tf.FixedLenFeature(
            [],
            tf.string),
        'data/last_frame': tf.FixedLenFeature(
            [],
            tf.string),
        'data/intermediate_frames': tf.FixedLenFeature(
            [],
            tf.string),
        'data/meta_file_names': tf.FixedLenFeature(
            [],
            tf.string)}

    parsed = tf.parse_single_example(
        ser,
        features=keys_to_features)

    fFrame = tf.decode_raw(
        parsed['data/first_frame'],
        tf.uint8)

    lFrame = tf.decode_raw(
        parsed['data/last_frame'],
        tf.uint8)

    iFrame = tf.decode_raw(
        parsed['data/intermediate_frames'],
        tf.uint8)

    meta_file_names = parsed['data/meta_file_names']

    # reshape images
    fFrame = tf.reshape(
        fFrame,
        [height, width, 1])

    lFrame = tf.reshape(
        lFrame,
        [height, width, 1])

    iFrame = tf.reshape(
        iFrame,
        [n_intermediate_frames, height, width, 1])

    # check flag for augmentations
    if is_training:
        fFrame, lFrame, iFrame = tf_augmentations.augment(
            fFrame,
            lFrame,
            iFrame)

    # cast images to float
    fFrame = tf.cast(fFrame, tf.float32)
    lFrame = tf.cast(lFrame, tf.float32)
    iFrame = tf.cast(iFrame, tf.float32)

    # pixels in range [-1, 1]
    fFrame = fFrame / 127.5 - 1.
    lFrame = lFrame / 127.5 - 1.
    iFrame = iFrame / 127.5 - 1.
    
    if is_training:
        fFrames, lFrames, iFrames, mfn = tf.train.shuffle_batch(
            [fFrame, lFrame, iFrame, meta_file_names],
            batch_size=batch_size,
            capacity=1000000,
            min_after_dequeue=10000,
            allow_smaller_final_batch=False,
            num_threads=4)

    else:
        fFrames, lFrames, iFrames, mfn = tf.train.batch(
            [fFrame, lFrame, iFrame, meta_file_names],
            batch_size=batch_size,
            capacity=1000,
            allow_smaller_final_batch=False,
            num_threads=2)

    return fFrames, lFrames, iFrames, mfn


def unit_test():
    current_path = os.path.join(
        '/neuhaus/movie/dataset',
        'tf_records',
        'slack_20px_fluorescent_window_5')

    filename = 'train.tfrecords'
    final_path = os.path.join(
        current_path,
        filename)

    with tf.Session().as_default() as sess:
        train_queue = tf.train.string_input_producer(
            [final_path], num_epochs=None)
        fFrames, lFrames, iFrames, mfns = read_and_decode(
            filename_queue=train_queue,
            is_training=True)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        import time
        start = time.time()
        for i in range(200):
            fF, lF, iF, mFN = sess.run(
                [
                    fFrames,
                    lFrames,
                    iFrames,
                    mfns])

            print(fF.shape, lF.shape, iF.shape) #, mFN) 

        print('Time taken:{} seconds.....'.format(
            str(
                round(
                    time.time() - start,
                    3))))

        coord.request_stop()

# unit_test()
