import os

import tensorflow as tf
import tf_augmentations

def read_and_decode(filename_queue=[], is_training=False,
                    batch_size=32, height=100, width=100,
                    n_intermediate_frames=3):

    reader = tf.RecordReader()
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
        [height, width])

    lFrame = tf.reshape(
        lFrame,
        [height, width])

    iFrame = tf.reshape(
        iFrame,
        [n_intermediate_frames, height, width])

    # check flag for augmentations
    if is_training:
        fFrame, lFrame = tf_augmentations.augment(
            fFrame,
            lFrame,
            iFrame)

    # cast images to float
    fFrame = tf.cast(
        fFrame,
        tf.float32)
    lFrame = tf.cast(
        lFrame,
        tf.float32)
    iFrame = tf.cast(
        iFrame,
        tf.float32)

    # pixels in range [-1, 1]
    fFrame = fFrame / 127.5 - 1.
    lFrame = lFrame / 127.5 - 1.
    iFrame = iFrame / 127.5 - 1.

    if is_training:
        fFrames, lFrames, iFrames = tf.train.shuffle_batch(
            [fFrame, lFrame, iFrame],
            batch_size=batch_size,
            capacity=1000,
            min_after_dequeue=500,
            allow_smaller_final_batch=False,
            num_threads=2)

    else:
        fFrames, lFrames, iFrames = tf.train.batch(
            [fFrame, lFrame, iFrame],
            batch_size=batch_size,
            capacity=1000,
            allow_smaller_final_batch=False,
            num_threads=2)

    return fFrames, lFrames, iFrames


def unit_test()
    current_path = os.path.join(
        '/neuhaus/movie/dataset',
        'tf_records',
        'slack_20px_fluorescent_window')

    filename = 'train.tfrecords'
    final_path = os.path.join(
        current_path,
        filename)

    with tf.Session().as_default() as sess:
        train_queue = tf.train.string_input_producer(
            [final_path], num_epochs=None)
        fFrames, lFrames, iFrames = read_and_decode(
            filename_queue=train_queue,
            is_training=False)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        for i in range(100):
            fF, lF, iF = sess.run([
                fFrames, lFrames, iFrames])

            print(fF.shape, lF.shape, iF,shape) 

unit_test()
