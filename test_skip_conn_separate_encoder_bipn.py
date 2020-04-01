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
from models.utils.optimizer import count_parameters
from models.utils.losses import huber_loss
from models.utils.losses import l2_loss
from models.utils.visualizer import visualize_frames
from models.utils.metrics import *

from models import skip_separate_encoder_bipn
from models import skip_unet_separate_encoder_bipn
from models import vgg16

ROOT_DIR = '/media/data/movie/dataset/tf_records/'

def get_files():
    runnable = []
    experiments = os.listdir(ROOT_DIR)
    for exp in experiments:
        models = os.listdir(ROOT_DIR + exp + '/')

        for model in models:
            if not model.endswith('pkl') and\
                not model.endswith('tfrecords'):

                model_path = os.path.join(
                    ROOT_DIR,
                    exp + '/',
                    model)

                if not os.path.exists(model_path +\
                    'evaluation.pkl'):
                    runnable.append(model_path)
                
    return runnable


def get_model_details(model_path):
    model = model_path.split('/')[-1]
    details = model.split('_')

    model_name = details[0]
    loss_index = details.index('l2')
    nIF = int(details[loss_index + 1][-1])
    out_channels = int(details[loss_index + 2].split(
        '-')[-1])
    additional_info = details[-1]

    return model_name, nIF, out_channels,\
        additional_info
            

def testing(model_path, args):
    
    splits = model_path.split('/')[:-1]
    TEST_REC_PATH = os.path.join(
        '/'.join(splits), 
        'test.tfrecords')
    
    # Get the latest checkpoint path
    weight_path = os.listdir(model_path)
    weight_path = [
        i
        for i in weight_path
        if '99000' in i and 'meta' in i][0]
    weight_path = weight_path[:-5]
    weight_path = os.path.join(
        model_path,
        weight_path)
    
    # get model details
    model_name, nIF, out_channels, additional_info = \
        get_model_details(model_path)

    # get #test_samples based on experiment
    if n_IF == 3: test_samples = 18300
    elif n_IF == 4: test_samples = 18296
    elif n_IF == 5: test_samples = 18265
    elif n_IF == 6: test_samples = 18235
    elif n_IF == 7: test_samples = 18204
    test_iters = test_samples // args.batch_size

    # SCOPING BEGINS HERE
    tf.reset_default_graph()
    with tf.Session() as sess:
        global_step = tf.train.get_global_step()

        test_queue = tf.train.string_input_producer(
            [TEST_REC_PATH], num_epochs=1)
        test_fFrames, test_lFrames, test_iFrames, test_mfn =\
            read_and_decode(
                filename_queue=test_queue,
                is_training=False,
                batch_size=args.batch_size,
                n_intermediate_frames=n_IF)

        with tf.variable_scope('separate_bipn'):
            print('TEST FRAMES (first):')
            if model_name == 'skip':
                test_rec_iFrames = skip_separate_encoder_bipn.build_bipn(
                    test_fFrames,
                    test_lFrames,
                    use_batch_norm=True,
                    is_training=False,
                    n_IF=n_IF,
                    starting_out_channels=out_channels)
            elif model_name == 'unet':
                test_rec_iFrames = skip_unet_separate_encoder_bipn.build_bipn(
                    test_fFrames,
                    test_lFrames,
                    use_batch_norm=True,
                    is_training=False,
                    n_IF=n_IF,
                    starting_out_channels=out_channels)

        print('Global parameters:{}'.format(
            count_parameters(tf.global_variables())))
        print('Learnable model parameters:{}'.format(
            count_parameters(tf.trainable_variables())))

        # DEFINE LOSS 
        if args.loss_id == 0:
            test_loss = huber_loss(
                test_iFrames, test_rec_iFrames,
                delta=1.)

        elif args.loss_id == 1:
            test_loss = l2_loss(
                test_iFrames, test_rec_iFrames)

        # DEFINE METRICS
        repeat_fFrame = metric_repeat_fframe(
            test_fFrames,
            test_iFrames)
        repeat_lFrame = metric_repeat_lframe(
            test_lFrames,
            test_iFrames)
        weighted_frame = metric_weighted_frame(
            test_fFrames,
            test_iFrames,
            test_lFrames)
        inter_frame = metric_interpolated_frame(
            test_iFrames,
            test_rec_iFrames)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        saver = tf.train.Saver()

        sess.run(init_op)

        # Load checkpoints
        sess.restore(sess, weight_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        metrics = {}
        metrics['repeat_first'] = []
        metrics['repeat_last'] = []
        metrics['weighted_frames'] = []
        metrics['inter_frames'] = []

        # START TRAINING HERE
        for iteration in range(test_iters):

            # get metrics
            repeat_first, repeat_last, weighted, true_metric = sess.run(
                [repeat_fFrame, repeat_lFrame, weighted_frame,\
                    inter_frame])
            metrics['repeat_first'].append(repeat_first)
            metrics['repeat_last'].append(repeat_last)
            metrics['weighted_frames'].append(weighted)
            metrics['inter_frames'].append(true_metric)

            # get plots
            start_frames, end_frames, mid_frames,\
                rec_mid_frames = sess.run(
                    [test_fFrames, test_lFrames,\
                        test_iFrames,
                        test_rec_iFrames])

            visualize_frames(
                start_frames,
                end_frames,
                mid_frames,
                rec_mid_frames,
                training=False,
                iteration=iteration,
                save_path=os.path.join(
                    model_path 
                    'test_plots/'))

        
        print('Testing complete.....')
        with open(model_path + '/evaluation.pkl')\
            as handle:
            pickle.dump(metrics, handle)
        print('Pickle file dumped.....')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--test_iters',
        type=int,
        default=15000,
        help='Mention the number of training iterations')

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
        help='0:huber, 1:l2')

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

    args = parser.parse_args()

    if args.optimizer == 'adam': args.optim_id = 1
    elif args.optimizer == 'sgd': args.optim_id = 2

    if args.loss == 'huber': args.loss_id = 0
    elif args.loss == 'l2': args.loss_id = 1

    # ckpt_folder_name: model-name_iters_batch_size_\
    # optimizer_lr_main-loss_starting-out-channels_\
    # additional-losses_loss-reg
    args.ckpt_folder_name = '{}_{}_{}_{}_{}_{}_nIF-{}_startOutChannels-{}'.format(
        args.model_name,
        str(100000),
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

    if args.debug:
        args.ckpt_folder_name = 'demo'

    training(args)

