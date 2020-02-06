import pickle
import cv2
import numpy as np
import os
import random
import argparse

import tensorflow as tf


def get_data(IMAGE_DIR, window):
    
    data = []

    folders = os.listdir(
        IMAGE_DIR)

    for folder in folders:
        folder_path = os.path.join(
            IMAGE_DIR,
            folder)

        images = os.listdir(
            folder_path)

        images = sorted(images)
        images = [
            os.path.join(
                folder_path,
                image)
            for image in images]

        for i in range(len(images) - window):
            data.append(images[i : i + window])

    return data

def get_splits(data, val_split, test_split):

    train_split = 1.0 - (val_split + test_split)

    random.shuffle(data)
    dataLength = len(data)

    train_data = data[
        0 : int(dataLength * train_split)]

    val_data = data[
        int(dataLength * train_split) :\
        int(dataLength * (train_split + val_split))]

    test_data = data[
        int(dataLength * (train_split + val_split)):]

    return train_data, val_data, test_data

##### TF helper functions
def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value]))

def feat_example(fFrame, lFrame, iFrames, metaFileNames):
    feature = {
        'data/first_frame': _bytes_feature(
            tf.compat.as_bytes(
                fFrame.tostring())),
        'data/last_frame': _bytes_feature(
            tf.compat.as_bytes(
                lFrame.tostring())),
        'data/intermediate_frames': _bytes_feature(
            tf.compat.as_bytes(
                iFrames.tostring())),
        'data/meta_file_names': _bytes_feature(
            tf.compat.as_bytes(
                metaFileNames))}

    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature))

    return example

def read_image(filename):
    return cv2.imread(
        filename, 0)

def dump_pickle(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(
            data,
            handle)

def write_tfr(TFR_DIR, data):

    writer = tf.python_io.TFRecordWriter(
        TFR_DIR)

    for tup_id in range(len(data)):
        tup = data[tup_id]
        metaFileNames = ', '.join(tup)
        fFrame, lFrame = tup[0], tup[-1]
        iFrames = tup[1: -1]
 
        fFrame = read_image(fFrame)
        lFrame = read_image(lFrame)
        
        height, width = fFrame.shape

        # Stack intermediate frames
        intermediateFrames = np.zeros(
            (len(iFrames), height, width),
            dtype=np.uint8)

        for frame_id in range(len(iFrames)):
            intermediateFrames[frame_id] = read_image(
                iFrames[frame_id])

        example = feat_example(
            fFrame,
            lFrame,
            intermediateFrames,
            metaFileNames)

        writer.write(
            example.SerializeToString())
    
        if tup_id % 5000 == 0:
            print('Wrote {}/{} examples.....'.format(
                tup_id, len(data)))

    writer.close()


def control(args):
    
    TFR_DIR = os.path.join(
        args.TFR_DIR,
        'slack_20px_fluorescent_window={}'.format(
            args.window))

    if not os.path.exists(TFR_DIR):
        os.makedirs(TFR_DIR)

    data = get_data(
        args.IMAGE_DIR,
        args.window)

    train_data, val_data, test_data = get_splits(
        data,
        args.VAL_SPLIT,
        args.TEST_SPLIT)

    # DUMP pickle files
    dump_pickle(
        TFR_DIR + '/train_meta_files.pkl',
        train_data)
    dump_pickle(
        TFR_DIR + '/validation_meta_files.pkl',
        val_data)
    dump_pickle(
        TFR_DIR + '/test_meta_files.pkl',
        test_data)
    print('Meta files dumped.....')

    # Write TF Records
    print('Splits created.....Writing TFRecords.....')
    print('Writing train TFR.....')
    write_tfr(
        TFR_DIR + '/train.tfrecords',
        train_data)
    print('Finished writing train TFR.....')
    print('Writing validation TFR.....')
    write_tfr(
        TFR_DIR + '/val.tfrecords',
        val_data)
    print('Finished writing validation TFR.....')
    print('Writing test TFR.....')
    write_tfr(
        TFR_DIR + '/test.tfrecords',
        test_data)
    print('Finished writing test TFR.....') 
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--IMAGE_DIR',
        type=str,
        default=os.path.join(
            '/neuhaus/movie/dataset',
            'slack_20px',
            'fluorescent'),
        help='path where images are present')

    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='mentions the number of frames in each\
            batch. 1 frame corresponds to 6 seconds')

    parser.add_argument(
        '--VAL_SPLIT',
        type=float,
        default=0.15,
        help='specifies the percentage of data to be\
            used for validation')

    parser.add_argument(
        '--TEST_SPLIT',
        type=float,
        default=0.15,
        help='specifies the percentage of data to be\
            used for testing')

    parser.add_argument(
        '--TFR_DIR',
        type=str,
        default=os.path.join(
            '/neuhaus/movie/dataset',
            'tf_records'),
        help='root path where TF Records will be saved')

    args = parser.parse_args()

    control(args)
