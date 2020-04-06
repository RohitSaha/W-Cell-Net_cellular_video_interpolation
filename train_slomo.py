import os
import pickle
import numpy as np
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
from tensorflow.contrib import summary

from data_pipeline.read_record import read_and_decode

from utils.optimizer import get_optimizer
from utils.optimizer import count_parameters
from utils.visualizer import visualize_frames

from models import slomo
from models import vgg16


def training(args):
    
    # DIRECTORY FOR CKPTS and META FILES
    # ROOT_DIR = '/neuhaus/movie/dataset/tf_records'
    ROOT_DIR = '/media/data/movie/dataset/tf_records'
    TRAIN_REC_PATH = os.path.join(
        ROOT_DIR,
        args.experiment_name,
        'train.tfrecords')
    VAL_REC_PATH = os.path.join(
        ROOT_DIR,
        args.experiment_name,
        'val.tfrecords')
    CKPT_PATH = os.path.join(
        ROOT_DIR,
        args.experiment_name,
        args.ckpt_folder_name + '/')

    # SCOPING BEGINS HERE
    with tf.Session().as_default() as sess:

        train_queue = tf.train.string_input_producer(
            [TRAIN_REC_PATH], num_epochs=None)
        train_fFrames, train_lFrames, train_iFrames, train_mfn =\
            read_and_decode(
                filename_queue=train_queue,
                is_training=True,
                batch_size=args.batch_size)

        val_queue = tf.train.string_input_producer(
            [VAL_REC_PATH], num_epochs=None)
        val_fFrames, val_lFrames, val_iFrames, val_mfn = \
            read_and_decode(
                filename_queue=val_queue,
                is_training=False,
                batch_size=args.batch_size)

        with tf.variable_scope('slomo'):
            print('TRAIN FRAMES (first):')
            train_output = slomo.SloMo_model(train_fFrames,
                train_lFrames,first_kernel=7,
                second_kernel=5,reuse=False,
                t_steps=3,verbose=False)

            train_rec_iFrames = train_output[0]

            train_flow_01 = train_output[1]
            train_flow_10 = train_output[2]
            train_weighted_ft0 = train_output[3]
            train_weighted_ft1 = train_output[4]

        with tf.variable_scope('slomo', reuse=tf.AUTO_REUSE):
            print('VAL FRAMES (first):') 
            val_output = slomo.SloMo_model(val_fFrames,
                val_lFrames,first_kernel=7,
                second_kernel=5,reuse=False,
                t_steps=3,verbose=False)

            val_rec_iFrames = val_output[0]
            val_flow_01 = val_output[1]
            val_flow_10 = val_output[2]
            val_weighted_ft0 = val_output[3]
            val_weighted_ft1 = val_output[4]


        # Weights should be kept locally ~ 500 MB space
        with tf.variable_scope('vgg16'):
            train_iFrames_features = vgg16.build_vgg16(
                train_iFrames, end_point='pool5').features
        with tf.variable_scope('vgg16', reuse=tf.AUTO_REUSE):
            train_rec_iFrames_features = vgg16.build_vgg16(
                train_rec_iFrames, end_point='pool5').features

        print('Global parameters:{}'.format(
            count_parameters(tf.global_variables())))
        print('Learnable model parameters:{}'.format(
            count_parameters(tf.trainable_variables())))

        train_l2_loss = slomo.l2_loss(train_iFrames,train_rec_iFrames)

        percep_loss = slomo.l2_loss(
            train_iFrames_features,
            train_rec_iFrames_features)

        wrap_loss = slomo.wrapping_loss(train_fFrames,train_lFrames,
            train_iFrames,train_flow_01,train_flow_10, 
            train_weighted_ft0, train_weighted_ft1)
        
        smooth_loss = slomo.smoothness_loss(train_flow_01,
            train_flow_10)

        # DEFINE METRICS
        val_loss = slomo.l2_loss(
            val_iFrames, val_rec_iFrames)

        total_train_loss = 0.1*train_l2_loss+1.0*percep_loss+\
            1.0*wrap_loss+50.0*smooth_loss

        # SUMMARIES
        tf.summary.scalar('train_l2_loss', train_l2_loss)
        tf.summary.scalar('wrap_loss', wrap_loss)
        tf.summary.scalar('smooth_loss', smooth_loss)
        tf.summary.scalar('percep_loss', percep_loss)
        tf.summary.scalar('total_val_l2_loss', val_loss)
        tf.summary.scalar('total_train_loss', total_train_loss)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            CKPT_PATH + 'train',
            sess.graph)

        with tf.variable_scope("global_step_and_learning_rate"):
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                global_step,
                50000, #FLAGS.decay_step
                0.1, #FLAGS.decay_rate
                staircase=True) #FLAGS.stair
            incr_global_step = tf.assign(
                global_step,
                global_step + 1)

        with tf.variable_scope("optimizer"):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                tvars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='slomo')
                optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=0.9)
                grads_and_vars = optimizer.compute_gradients(
                    total_train_loss,
                    tvars)
                train_op = optimizer.apply_gradients(
                    grads_and_vars)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        saver = tf.train.Saver()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        # START TRAINING HERE
        for iteration in range(args.train_iters):
            _, t_summ, t_loss = sess.run(
                [train_op, merged, total_train_loss])

            train_writer.add_summary(t_summ, iteration)
            print('Iter:{}/{}, Train Loss:{}'.format(
                iteration,
                args.train_iters,
                t_loss))

            if iteration % args.val_every == 0:
                v_loss = sess.run(val_loss)
                print('Iter:{}, Val Loss:{}'.format(
                    iteration,
                    v_loss))

            if iteration % args.save_every == 0:
                saver.save(
                    sess,
                    CKPT_PATH + 'iter:{}_val:{}'.format(
                        str(iteration),
                        str(round(v_loss, 3))))

            if iteration % args.plot_every == 0:
                start_frames, end_frames, mid_frames,\
                    rec_mid_frames = sess.run(
                        [train_fFrames, train_lFrames,\
                            train_iFrames,\
                            train_rec_iFrames])

                visualize_frames(
                    start_frames,
                    end_frames,
                    mid_frames,
                    rec_mid_frames,
                    training=True,
                    iteration=iteration,
                    save_path=os.path.join(
                        CKPT_PATH,
                        'train_plots/'))

                start_frames, end_frames, mid_frames,\
                    rec_mid_frames = sess.run(
                        [val_fFrames, val_lFrames,\
                            val_iFrames,
                            val_rec_iFrames])

                visualize_frames(
                    start_frames,
                    end_frames,
                    mid_frames,
                    rec_mid_frames,
                    training=False,
                    iteration=iteration,
                    save_path=os.path.join(
                        CKPT_PATH,
                        'validation_plots/'))

        print('Training complete.....')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--train_iters',
        type=int,
        default=100000,
        help='Mention the number of training iterations')

    parser.add_argument(
        '--val_every',
        type=int,
        default=100,
        help='Number of iterations after which validation is done')

    parser.add_argument(
        '--save_every',
        type=int,
        default=5000,
        help='Number of iterations after which model is saved')

    parser.add_argument(
        '--plot_every',
        type=int,
        default=1000,
        help='Nu,ber of iterations after which plots will be saved')

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='slack_20px_fluorescent_window_5',
        help='to mention the experiment folder in tf_records')

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='1. adam, 2. SGD + momentum')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='To mention the starting learning rate')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='To mention the number of samples in a batch')

    parser.add_argument(
        '--loss',
        type=str,
        default='l2',
        help='0:l1, 1:l2')

    parser.add_argument(
        '--perceptual_loss_weight',
        type=int,
        default=0,
        help='Mention strength of perceptual loss')

    parser.add_argument(
        '--perceptual_loss_endpoint',
        type=str,
        default='conv4_3',
        help='Mentions the layer from which features are to be extracted')

    parser.add_argument(
        '--model_name',
        type=str,
        default='slowmo',
        help='Mentions name of model to be run')

    args = parser.parse_args()

    if args.optimizer == 'adam': args.optim_id = 1
    elif args.optimizer == 'sgd': args.optim_id = 2

    if args.loss == 'l1': args.loss_id = 0
    elif args.loss == 'l2': args.loss_id = 1

    # ckpt_folder_name: model-name_iters_batch_size_\
    # optimizer_lr_main-loss_additional-losses_loss-reg
    args.ckpt_folder_name = '{}_{}_{}_{}_{}_{}'.format(
        args.model_name,
        str(args.train_iters),
        str(args.batch_size),
        args.optimizer,
        str(args.learning_rate),
        args.loss)

    if args.perceptual_loss_weight:
        args.ckpt_folder_name += '_{}-{}_{}'.format(
            'perceptualLoss',
            args.perceptual_loss_endpoint,
            str(args.perceptual_loss_weight))

    training(args)

