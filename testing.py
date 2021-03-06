import os
import pickle
import numpy as np
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib import summary

from data_pipeline.read_record import read_and_decode

from utils.optimizer import count_parameters
from utils.losses import huber_loss
from utils.losses import l2_loss
from utils.visualizer import visualize_frames
from utils.metrics import metric_repeat_fframe
from utils.metrics import metric_repeat_lframe
from utils.metrics import metric_weighted_frame
from utils.metrics import metric_interpolated_frame

from models import wnet
from models import slomo
from models import BiPN
from models import vgg16

def testing(info):
    
    # Get the best checkpoint path
    weight_path = os.listdir(
        info['model_path'])

    weight_paths = [
        i
        for i in weight_path
        if 'meta' in i]
    di_weight = {}
    for path in weight_paths:
        di_weight[path] = int(path.split(':')[-1].split('.')[0])
    di_weight = {
        k: v
        for k, v in sorted(di_weight.items(), key=lambda item: item[1])}
    weight_path = [*di_weight][0]

    weight_path = weight_path[:-5]
    weight_path = os.path.join(
        info['model_path'],
        weight_path)

    n_IF = info['n_IF']
    batch_size = info['batch_size']
    # get #test_samples based on experiment
    if n_IF == 3: test_samples = 18300
    elif n_IF == 4: test_samples = 18296
    elif n_IF == 5: test_samples = 18265
    elif n_IF == 6: test_samples = 18235
    elif n_IF == 7: test_samples = 18204
    test_iters = test_samples // batch_size
    test_samples = test_samples - (test_samples % batch_size)

    # get attention
    if info['attention']:
        use_attention = 1
        if info['use_spatial_attention']:
            spatial_attention = 1
        else:
            spatial_attention = 0
    else:
        use_attention = 0
        spatial_attention = 0

    # SCOPING BEGINS HERE
    tf.reset_default_graph()
    with tf.Session() as sess:
        global_step = tf.train.get_global_step()

        test_queue = tf.train.string_input_producer(
            [info['TEST_REC_PATH']], num_epochs=1)
        test_fFrames, test_lFrames, test_iFrames, test_mfn =\
            read_and_decode(
                filename_queue=test_queue,
                is_training=False,
                batch_size=batch_size,
                n_intermediate_frames=n_IF,
                allow_smaller_final_batch=False)

        if info['model_name'] in ['skip', 'wnet']:
            with tf.variable_scope('separate_bipn'):
                print('TEST FRAMES (first):')
                if info['model_name'] == 'skip':
                    test_rec_iFrames = skip_separate_encoder_bipn.build_bipn(
                        test_fFrames,
                        test_lFrames,
                        use_batch_norm=True,
                        is_training=False,
                        n_IF=n_IF,
                        starting_out_channels=info['out_channels'],
                        use_attention=use_attention,
                        spatial_attention=spatial_attention,
                        is_verbose=False)

                elif info['model_name'] == 'wnet':
                    test_rec_iFrames = wnet.build_wnet(
                        test_fFrames,
                        test_lFrames,
                        use_batch_norm=True,
                        is_training=False,
                        n_IF=n_IF,
                        starting_out_channels=info['out_channels'],
                        use_attention=use_attention,
                        spatial_attention=spatial_attention,
                        is_verbose=False)

        elif info['model_name'] == 'slomo':
            with tf.variable_scope('slomo'):
                test_output = slomo.SloMo_model(
                    test_fFrames,
                    test_lFrames,
                    first_kernel=7,
                    second_kernel=5,
                    reuse=False,
                    t_steps=n_IF,
                    verbose=False)
                test_rec_iFrames = test_output[0]

        elif info['model_name'] == 'bipn':
            with tf.variable_scope('bipn'):
                test_rec_iFrames = BiPN.build_bipn(
                    test_fFrames,
                    test_lFrames,
                    n_IF=n_IF,
                    use_batch_norm=True,
                    is_training=False)

        print('Global parameters:{}'.format(
            count_parameters(tf.global_variables())))
        print('Learnable model parameters:{}'.format(
            count_parameters(tf.trainable_variables())))

        # DEFINE LOSS 
        if info['loss'] == 'l2':
            test_loss = l2_loss(
                test_iFrames, test_rec_iFrames)
        elif info['loss'] == 'l1':
            test_loss = l1_loss(
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
        saver.restore(sess, weight_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        metrics = {}
        metrics['learnable_parameters'] = count_parameters(tf.trainable_variables())
        metrics['repeat_first'] = []
        metrics['repeat_last'] = []
        metrics['weighted_frames'] = []
        metrics['inter_frames'] = []
        metrics['repeat_first_psnr'] = []
        metrics['repeat_last_psnr'] = []
        metrics['weighted_frames_psnr'] = []
        metrics['inter_frames_psnr'] = []

        print('EVALUATING:{}--------------------------->'.format(
            info['model_path']))

        # START TRAINING HERE
        for iteration in range(test_iters):

            # get frames and metrics
            start_frames, end_frames, mid_frames, rec_mid_frames,\
            repeat_first, repeat_last, weighted, true_metric = sess.run(
                [test_fFrames, test_lFrames, test_iFrames, test_rec_iFrames,\
                repeat_fFrame, repeat_lFrame, weighted_frame,\
                    inter_frame])

            samples = start_frames.shape[0]
            metrics['repeat_first'].append(repeat_first[0] * samples)
            metrics['repeat_last'].append(repeat_last[0] * samples)
            metrics['weighted_frames'].append(weighted[0] * samples)
            metrics['inter_frames'].append(true_metric[0] * samples)
            metrics['repeat_first_psnr'].append(repeat_first[1] * samples)
            metrics['repeat_last_psnr'].append(repeat_last[1] * samples)
            metrics['weighted_frames_psnr'].append(weighted[1] * samples)
            metrics['inter_frames_psnr'].append(true_metric[1] * samples)

            visualize_frames(
                start_frames,
                end_frames,
                mid_frames,
                rec_mid_frames,
                training=False,
                iteration=iteration,
                save_path=os.path.join(
                    info['model_path'], 
                    'test_plots' + '/'))

            if iteration % 50 == 0:
                print('{}/{} iters complete'.format(
                    iteration, test_iters))

        print('Testing complete.....')
        

    # Calculate metrics:
    mean_rf = sum(metrics['repeat_first']) / test_samples
    mean_rl = sum(metrics['repeat_last']) / test_samples
    mean_wf = sum(metrics['weighted_frames']) / test_samples
    mean_if = sum(metrics['inter_frames']) / test_samples

    metrics['mean_repeat_first'] = mean_rf
    metrics['mean_repeat_last'] = mean_rl
    metrics['mean_weighted_frames'] = mean_wf
    metrics['mean_inter_frames'] = mean_if

    mean_rf_psnr = sum(metrics['repeat_first_psnr']) / test_samples
    mean_rl_psnr = sum(metrics['repeat_last_psnr']) / test_samples
    mean_wf_psnr = sum(metrics['weighted_frames_psnr']) / test_samples
    mean_if_psnr = sum(metrics['inter_frames_psnr']) / test_samples

    metrics['mean_psnr_repeat_first'] = mean_rf_psnr
    metrics['mean_psnr_repeat_last'] = mean_rl_psnr
    metrics['mean_psnr_weighted_frames'] = mean_wf_psnr
    metrics['mean_psnr_inter_frames'] = mean_if_psnr

    with open(info['model_path'] + '/evaluation.pkl', 'wb') as handle:
        pickle.dump(metrics, handle)

    print('Pickle file dumped.....')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='params of running an experiment')

    parser.add_argument(
        '--model_name',
        default='wnet',
        type=str,
        help='Mention the model to test')
    
    parser.add_argument(
        '--window_size',
        default=5,
        type=int,
        help='Mention the window size to be used')

    parser.add_argument(
        '--out_channels',
        default=8,
        type=int,
        help='Mention the out channels of first conv layer')

    args = parser.parse_args()

    ROOT_DIR = '/media/data/movie/dataset/tf_records/'
    exp_name = 'slack_20px_fluorescent_window_{}/'
    model = 'unet_separate_encoder_bipn_100000_32_adam_0.001_l2_nIF-{}_startOutChannels-{}'
    info = {}

    window_size = args.window_size
    out_channels = args.out_channels

    exp_name = exp_name.format(str(window_size))
    model = model.format(str(window_size - 2), str(out_channels))

    info['model_path'] = os.path.join(ROOT_DIR, exp_name, model + '/')
    info['model_name'] = args.model_name
    info['batch_size'] = 32
    info['loss'] = 'l2'
    info['n_IF'] = window_size - 2
    info['out_channels'] = out_channels
    info['attention'] = 0
    info['use_spatial_attention'] = 1
    info['TEST_REC_PATH'] = os.path.join(ROOT_DIR, exp_name, 'test.tfrecords')

    testing(info)

