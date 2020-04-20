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
from data_pipeline.tf_augmentations import gaussian_filter 

from utils.optimizer import get_optimizer
from utils.optimizer import count_parameters
from utils.losses import l2_loss
from utils.losses import l1_loss
from utils.losses import ssim_loss
from utils.losses import ridge_weight_decay
from utils.losses import perceptual_loss
from utils.visualizer import visualize_frames

from models import discriminator
from models import generator_unet 
from models import vgg16

std_dev = 1

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

        # Apply gaussian blurring manually
        '''
        train_fFrames = gaussian_filter(train_fFrames, std=std_dev)
        train_lFrames = gaussian_filter(train_lFrames, std=std_dev)
        train_iFrames = gaussian_filter(train_iFrames, std=std_dev)
        val_fFrames = gaussian_filter(val_fFrames, std=std_dev)
        val_lFrames = gaussian_filter(val_lFrames, std=std_dev)
        val_iFrames = gaussian_filter(val_iFrames, std=std_dev)
        '''

        # TRAINABLE
        print('---------------------------------------------')
        print('----------------- GENERATOR -----------------')
        print('---------------------------------------------')
        with tf.variable_scope('generator'):
            train_rec_iFrames = generator_unet.build_generator(
                train_fFrames,
                train_lFrames,
                use_batch_norm=True,
                is_training=True,
                n_IF=args.n_IF,
                starting_out_channels=args.starting_out_channels,
                use_attention=args.use_attention,
                spatial_attention=args.spatial_attention,
                is_verbose=True)

        print('---------------------------------------------')
        print('-------------- DISCRIMINATOR ----------------')
        print('---------------------------------------------')
        # discriminator for classifying real images
        with tf.variable_scope('discriminator'):
            train_real_output_discriminator = discriminator.build_discriminator(
                train_iFrames,
                use_batch_norm=True,
                is_training=True,
                starting_out_channels=args.discri_starting_out_channels,
                is_verbose=True)
        # discriminator for classifying fake images
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            train_fake_output_discriminator = discriminator.build_discriminator(
                train_rec_iFrames,
                use_batch_norm=True,
                is_training=True,
                starting_out_channels=args.discri_starting_out_channels,
                is_verbose=False)

        # VALIDATION
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            val_rec_iFrames = generator_unet.build_generator(
                val_fFrames,
                val_lFrames,
                use_batch_norm=True,
                n_IF=args.n_IF,
                is_training=False,
                starting_out_channels=args.starting_out_channels,
                use_attention=args.use_attention,
                spatial_attention=args.spatial_attention,
                is_verbose=False)

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            val_real_output_discriminator = discriminator.build_discriminator(
                val_iFrames,
                use_batch_norm=True,
                is_training=False,
                starting_out_channels=args.discri_starting_out_channels,
                is_verbose=False)
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            val_fake_output_discriminator = discriminator.build_discriminator(
                val_rec_iFrames,
                use_batch_norm=True,
                is_training=False,
                starting_out_channels=args.discri_starting_out_channels,
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

        # DEFINE GAN losses:
        train_discri_real_loss = tf.reduce_sum(
            tf.square(
                train_real_output_discriminator - 1)) / (2 * args.batch_size)
        train_discri_fake_loss = tf.reduce_sum(
            tf.square(
                train_fake_output_discriminator)) / (2 * args.batch_size)
        train_discriminator_loss = train_discri_real_loss + train_discri_fake_loss

        train_generator_fake_loss = tf.reduce_sum(
            tf.square(
                train_fake_output_discriminator - 1)) / args.batch_size
        train_reconstruction_loss = l2_loss(
            train_rec_iFrames, train_iFrames) * args.reconstruction_loss_weight
        train_generator_loss = train_generator_fake_loss + train_reconstruction_loss

        val_discri_real_loss = tf.reduce_sum(
            tf.square(
                val_real_output_discriminator - 1)) / (2 * args.batch_size)
        val_discri_fake_loss = tf.reduce_sum(
            tf.square(
                val_fake_output_discriminator)) / (2 * args.batch_size)
        val_discriminator_loss = val_discri_real_loss + val_discri_fake_loss

        val_generator_fake_loss = tf.reduce_sum(
            tf.square(
                val_fake_output_discriminator - 1)) / args.batch_size
        val_reconstruction_loss = l2_loss(
            val_rec_iFrames, val_iFrames) * args.reconstruction_loss_weight
        val_generator_loss = val_generator_fake_loss + val_reconstruction_loss

        if args.perceptual_loss_weight:
            train_percp_loss = perceptual_loss(
                train_rec_iFrames_features, train_iFrames_features)
            train_generator_loss += args.perceptual_loss_weight * train_percp_loss

        # SUMMARIES
        tf.summary.scalar('train_discri_real_loss', train_discri_real_loss)
        tf.summary.scalar('train_discri_fake_loss', train_discri_fake_loss)
        tf.summary.scalar('train_discriminator_loss', train_discriminator_loss)
        tf.summary.scalar('train_generator_fake_loss', train_generator_fake_loss)
        tf.summary.scalar('train_reconstruction_loss', train_reconstruction_loss)
        tf.summary.scalar('train_generator_loss', train_generator_loss)

        tf.summary.scalar('val_discri_real_loss', val_discri_real_loss)
        tf.summary.scalar('val_discri_fake_loss', val_discri_fake_loss)
        tf.summary.scalar('val_discriminator_loss', val_discriminator_loss)
        tf.summary.scalar('val_generator_fake_loss', val_generator_fake_loss)
        tf.summary.scalar('val_reconstruction_loss', val_reconstruction_loss)
        tf.summary.scalar('val_generator_loss', val_generator_loss)       

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            CKPT_PATH + 'train',
            sess.graph)

        # get variables responsible for generator and discriminator
        trainable_vars = tf.trainable_variables()
        generator_vars = [
            var
            for var in trainable_vars
            if 'generator' in var.name]
        discriminator_vars = [
            var
            for var in trainable_vars
            if 'discriminator' in var.name]

        # DEFINE OPTIMIZERS
        generator_optimizer = get_optimizer(
            train_generator_loss,
            optim_id=args.optim_id,
            learning_rate=args.learning_rate,
            use_batch_norm=True,
            var_list=generator_vars)
        discriminator_optimizer = get_optimizer(
            train_discriminator_loss,
            optim_id=args.optim_id,
            learning_rate=args.learning_rate * 2.,
            use_batch_norm=True,
            var_list=discriminator_vars)

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

            for d_iteration in range(args.disc_train_iters):
                disc_, td_loss = sess.run(
                    [discriminator_optimizer, train_discriminator_loss])

            gene_, tgf_loss, tr_loss, t_summ = sess.run(
                [generator_optimizer, train_generator_fake_loss,\
                    train_reconstruction_loss, merged])

            train_writer.add_summary(t_summ, iteration)

            print('Iter:{}/{}, Disc. Loss:{}, Gen. Loss:{}, Rec. Loss:{}'.format(
                iteration,
                args.train_iters,
                str(round(td_loss, 6)),
                str(round(tgf_loss, 6)),
                str(round(tr_loss, 6))))

            if iteration % args.val_every == 0:
                vd_loss, vgf_loss, vr_loss = sess.run(
                    [val_discriminator_loss, val_generator_fake_loss,\
                        val_reconstruction_loss])
                print('Iter:{}, VAL Disc. Loss:{}, Gen. Loss:{}, Rec. Loss:{}'.format(
                    iteration,
                    str(round(vd_loss, 6)),
                    str(round(vgf_loss, 6)),
                    str(round(vr_loss, 6))))

            if iteration % args.save_every == 0:
                saver.save(
                    sess,
                    CKPT_PATH + 'iter:{}_valDisc:{}_valGen:{}_valRec:{}'.format(
                        str(iteration),
                        str(round(vd_loss, 6)),
                        str(round(vgf_loss, 6)),
                        str(round(vr_loss, 6))))

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
        '--disc_train_iters',
        type=int,
        default=1,
        help='Mention the number of nested iterations for discriminator')

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
        '--reconstruction_loss_weight',
        type=float,
        default=0.01,
        help='Mention strength of reconstruction loss')

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

    parser.add_argument(
        '--discri_starting_out_channels',
        type=int,
        default=8,
        help='Specifies the number of channels in first conv layer of discriminator')

    args = parser.parse_args()

    if args.optimizer == 'adam': args.optim_id = 1
    elif args.optimizer == 'sgd': args.optim_id = 2

    # ckpt_folder_name: model-name_iters_batch_size_\
    # optimizer_lr_starting-out-channels_\
    # additional-losses_loss-reg
    args.ckpt_folder_name = '{}_{}_{}_{}_{}_{}_nIF-{}_startOutChannels-{}'.format(
        args.model_name,
        str(args.train_iters),
        str(args.disc_train_iters),
        str(args.batch_size),
        args.optimizer,
        str(args.learning_rate),
        str(args.n_IF),
        str(args.starting_out_channels))

    args.ckpt_folder_name += '_reconstruction-weight-{}'.format(
        str(args.reconstruction_loss_weight))

    if args.perceptual_loss_weight:
        args.ckpt_folder_name += '_perceptualLoss-{}-{}'.format(
            args.perceptual_loss_endpoint,
            str(args.perceptual_loss_weight))

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

