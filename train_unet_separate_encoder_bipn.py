import os
import pickle
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib import summary

from data_pipeline.read_record import read_and_decode
from models.utils.optimizer import get_optimizer
from models.utils.optimizer import count_parameters
from models.utils.losses import huber_loss
from models.utils.losses import l2_loss
from models.utils.losses import l1_loss
from models.utils.losses import ssim_loss
from models.utils.losses import ridge_weight_decay
from models.utils.losses import perceptual_loss
from models.utils.visualizer import visualize_frames

from models import bipn
from models import separate_encoder_bipn
from models import skip_separate_encoder_bipn
from models import skip_unet_separate_encoder_bipn
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
        global_step = tf.train.get_global_step()

        train_queue = tf.train.string_input_producer(
            [TRAIN_REC_PATH], num_epochs=None)
        train_fFrames, train_lFrames, train_iFrames, train_mfn =\
            read_and_decode(
                filename_queue=train_queue,
                is_training=True,
                batch_size=args.batch_size,
                n_intermediate_frames=args.n_IF)

        val_queue = tf.train.string_input_producer(
            [VAL_REC_PATH], num_epochs=None)
        val_fFrames, val_lFrames, val_iFrames, val_mfn = \
            read_and_decode(
                filename_queue=val_queue,
                is_training=False,
                batch_size=args.batch_size,
                n_intermediate_frames=args.n_IF)

        with tf.variable_scope('separate_bipn'):
            print('TRAIN FRAMES (first):')
            train_rec_iFrames = skip_unet_separate_encoder_bipn.build_bipn(
                train_fFrames,
                train_lFrames,
                use_batch_norm=True,
                is_training=True,
                n_IF=args.n_IF,
                starting_out_channels=args.starting_out_channels,
                use_attention=args.use_attention,
                spatial_attention=args.spatial_attention,
                is_verbose=True)

        with tf.variable_scope('separate_bipn', reuse=tf.AUTO_REUSE):
            print('VAL FRAMES (first):')
            val_rec_iFrames = skip_unet_separate_encoder_bipn.build_bipn(
                val_fFrames,
                val_lFrames,
                use_batch_norm=True,
                is_training=False,
                n_IF=args.n_IF,
                starting_out_channels=args.starting_out_channels,
                use_attention=args.use_attention,
                spatial_attention=args.spatial_attention,
                is_verbose=False)
            
        if args.perceptual_loss_weight:
            # Weights should be kept locally ~ 500 MB space
            with tf.variable_scope('vgg16'):
                train_iFrames_features = vgg16.build_vgg16(
                    train_iFrames,
                    end_point=args.perceptual_loss_endpoint).features
            with tf.variable_scope('vgg16', reuse=tf.AUTO_REUSE):
                train_rec_iFrames_features = vgg16.build_vgg16(
                    train_rec_iFrames,
                    end_point=args.perceptual_loss_endpoint).features

        print('Global parameters:{}'.format(
            count_parameters(tf.global_variables())))
        print('Learnable model parameters:{}'.format(
            count_parameters(tf.trainable_variables())))

        # DEFINE METRICS
        if args.loss_id == 0:
            train_loss = huber_loss(
                train_iFrames, train_rec_iFrames,
                delta=1.)
            val_loss = huber_loss(
                val_iFrames, val_rec_iFrames,
                delta=1.)

        elif args.loss_id == 1:
            train_loss = l2_loss(
                train_iFrames, train_rec_iFrames)
            val_loss = l2_loss(
                val_iFrames, val_rec_iFrames) 

        elif args.loss_id == 2:
            train_loss = l1_loss(
                train_iFrames, train_rec_iFrames)
            val_loss = l1_loss(
                val_iFrames, val_rec_iFrames)
        
        elif args.loss_id == 3:
            train_loss = ssim_loss(
                train_rec_iFrames, train_iFrames)
            val_loss = ssim_loss(
                val_rec_iFrames, val_iFrames)

        total_train_loss = train_loss
        tf.summary.scalar('train_l2_loss', train_loss)
        tf.summary.scalar('total_val_l2_loss', val_loss)

        if args.perceptual_loss_weight:
            train_perceptual_loss = perceptual_loss(
                train_iFrames_features,
                train_rec_iFrames_features)

            tf.summary.scalar('train_perceptual_loss',\
                train_perceptual_loss)

            total_train_loss += train_perceptual_loss\
                * args.perceptual_loss_weight

        if args.weight_decay:
            decay_loss = ridge_weight_decay(
                tf.trainable_variables())

            tf.summary.scalar('ridge_l2_weight_decay',\
                decay_loss)

            total_train_loss += decay_loss\
                * args.weight_decay 

        # SUMMARIES
        tf.summary.scalar('total_train_loss',\
            total_train_loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            CKPT_PATH + 'train',
            sess.graph)

        # DEFINE OPTIMIZER
        optimizer = get_optimizer(
            train_loss,
            optim_id=args.optim_id,
            learning_rate=args.learning_rate,
            use_batch_norm=True)

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
                [optimizer, merged, total_train_loss])

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
        default=15000,
        help='Mention the number of training iterations')

    parser.add_argument(
        '--val_every',
        type=int,
        default=100,
        help='Number of iterations after which validation is done')

    parser.add_argument(
        '--save_every',
        type=int,
        default=100,
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
        default=1e-3,
        help='To mention the starting learning rate')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='To mention the number of samples in a batch')

    parser.add_argument(
        '--loss',
        type=str,
        default='l2',
        help='0:huber, 1:l2, 2:l1, 3:ssim')

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='To mention the strength of L2 weight decay')

    parser.add_argument(
        '--perceptual_loss_weight',
        type=float,
        default=1.0,
        help='Mention strength of perceptual loss')

    parser.add_argument(
        '--perceptual_loss_endpoint',
        type=str,
        default='conv4_3',
        help='Mentions the layer from which features are to be extracted')

    parser.add_argument(
        '--model_name',
        type=str,
        default='bipn',
        help='Mentions name of model to be run')

    parser.add_argument(
        '--starting_out_channels',
        type=int,
        default=8,
        help='Specify the number of out channels for the first conv')

    parser.add_argument(
        '--debug',
        type=int,
        default=1,
        help='Specifies whether to run the script in DEBUG mode')

    parser.add_argument(
        '--additional_info',
        type=str,
        default='',
        help='Additional details to identify model in dir')

    parser.add_argument(
        '--n_IF',
        type=int,
        default=3,
        help='Mentions the number of intermediate frames')

    parser.add_argument(
        '--use_attention',
        type=int,
        default=0,
        help='Specifies if self spatial attention is to be used')

    parser.add_argument(
        '--spatial_attention',
        type=int,
        default=0,
        help='Specifies whether to use spatial/channel attention')

    args = parser.parse_args()

    if args.optimizer == 'adam': args.optim_id = 1
    elif args.optimizer == 'sgd': args.optim_id = 2

    if args.loss == 'huber': args.loss_id = 0
    elif args.loss == 'l2': args.loss_id = 1
    elif args.loss == 'l1': args.loss_id = 2
    elif args.loss == 'ssim': args.loss_id = 3

    # ckpt_folder_name: model-name_iters_batch_size_\
    # optimizer_lr_main-loss_starting-out-channels_\
    # additional-losses_loss-reg
    args.ckpt_folder_name = '{}_{}_{}_{}_{}_{}_nIF-{}_startOutChannels-{}'.format(
        args.model_name,
        str(args.train_iters),
        str(args.batch_size),
        args.optimizer,
        str(args.learning_rate),
        args.loss,
        str(args.n_IF),
        str(args.starting_out_channels))

    if args.perceptual_loss_weight:
        args.ckpt_folder_name += '_perceptualLoss-{}-{}'.format(
            args.perceptual_loss_endpoint,
            str(args.perceptual_loss_weight))

    if args.weight_decay:
        args.ckpt_folder_name += '_ridgeWeightDecay-{}'.format(
            str(args.weight_decay))

    if args.use_attention:
        if args.spatial_attention:
            args.ckpt_folder_name += '_spatialAttention'
        else:
            args.ckpt_folder_name += '_channelAttention'

    if args.additional_info:
        args.ckpt_folder_name += '_{}'.format(
            args.additional_info)

    if args.debug:
        args.ckpt_folder_name = 'demo'

    training(args)

