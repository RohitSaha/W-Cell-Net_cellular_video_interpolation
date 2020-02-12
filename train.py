import os
import pickle
import argparse

import tensorflow as tf

from data_pipeline.read_record import read_and_decode
from models.utils.optimizer import get_optimizer
from models import bipn

def training(args):
    
    # DIRECTORY FOR CKPTS and META FILES


    # SCOPING BEGINS HERE
    with tf.Session().as_default() as sess:
        global_step = tf.train.get_global_step()

        train_queue = tf.train.string_input_producer(
            [args.TRAIN_REC_DIR], num_epochs=None)
        train_fFrames, train_lFrames, train_iFrames = read_and_decode(
            filename_queue=train_queue,
            is_traning=True)

        val_queue = tf.train.string_input_producer(
            [args.VAL_REC_DIR], num_epochs=None)
        val_fFrames, val_lFrames, val_iFrames = read_and_decode(
            filename_queue=val_queue,
            is_training=False)

        with tf.variable_scope('bipn'):
            train_rec_iFrames = bipn.build_bipn(
                train_fFrames,
                train_lFrames,
                use_batch_norm=True,
                is_training=True)

        with tf.variable_scope('bipn', reuse=tf.AUTO_REUSE):
            val_rec_iFrames = bipn.build_bipn(
                val_fFrames,
                val_lFrames,
                use_batch_norm=True,
                is_training=False)
            
        # DEFINE METRICS


        # DEFINE OPTIMIZER
        optimizer = get_optimizer(
            train_loss,
            optim_id=args.optim_id,
            learning_rate=args.learning_rate)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        # START TRAINING HERE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--optim_id',
        type=int,
        default=1,
        help='ID to specify the learning rate to be used')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='To mention the starting learning rate')

    args = parser.parse_args()

    training(args)

