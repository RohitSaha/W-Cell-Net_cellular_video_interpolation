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

from models import skip_separate_encoder_bipn
from models import skip_unet_separate_encoder_bipn
from models import slomo
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

            elif info['model_name'] == 'unet':
                test_rec_iFrames = skip_unet_separate_encoder_bipn.build_bipn(
                    test_fFrames,
                    test_lFrames,
                    use_batch_norm=True,
                    is_training=False,
                    n_IF=n_IF,
                    starting_out_channels=info['out_channels'],
                    use_attention=use_attention,
                    spatial_attention=spatial_attention,
                    is_verbose=False)

            elif info['model_name'] == 'slowmo':
                test_rec_iFrames = slomo.SloMo_model(
                    test_fFrames,
                    test_lFrames,
                    first_kernel=7,
                    second_kernel=5,
                    reuse=False,
                    t_steps=3,
                    verbose=False)

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
    # model = 'unet_separate_encoder_bipn_100000_32_adam_0.001_l2_nIF-{}_startOutChannels-{}'
    model = 'skip_conn_separate_encoder_bipn_100000_32_adam_0.001_l1_nIF-3_startOutChannels-8'
    info = {}

    window_size = args.window_size
    out_channels = args.out_channels

    exp_name = exp_name.format(str(window_size))
    # model = model.format(str(window_size - 2), str(out_channels))

    info['model_path'] = os.path.join(ROOT_DIR, exp_name, model + '/')
    info['model_name'] = 'skip'
    info['batch_size'] = 32
    info['loss'] = 'l2'
    info['n_IF'] = window_size - 2
    info['out_channels'] = out_channels
    info['attention'] = 0
    info['use_spatial_attention'] = 1
    info['TEST_REC_PATH'] = os.path.join(ROOT_DIR, exp_name, 'test.tfrecords')

    testing(info)

# LEGACY CODE:
def control(args):

    # get runnable files
    runnables = get_files()
    print('Experiments to run:')
    for runnable in runnables:
        print(runnable)

    master_metrics = {}
    best_rf = float('inf')
    best_rl = float('inf')
    best_wf = float('inf')
    best_if = float('inf')

    for model_path_id in range(len(runnables)):
        metrics, test_samples = testing(
            runnable[model_path_id],
            args)        

        rep_first = metrics['repeat_first']
        rep_last = metrics['repeat_last']
        weight_frames = metrics['weighted_frames']
        inter_frames = metrics['inter_frames']
    
        mean_rf = sum(rep_first) / test_samples
        mean_rl = sum(rep_last) / test_samples
        mean_wf = sum(weight_frames) / test_samples
        mean_if = sum(inter_frames) / test_samples

        master_metrics[runnable[model_path_id]] = [
            mean_rf, mean_rl, mean_wf, mean_if]

        # Get the best model out of all models
        if mean_rf < best_rf:
            best_rf = mean_rf
            best_rf_model = runnables[model_path_id]
        if mean_rl < best_rl:
            best_rl = mean_rl
            best_rl_model = runnables[model_path_id]
        if mean_wf < best_wf:
            best_wf = mean_wf
            best_wf_model = runnables[model_path_id]
        if mean_if < best_if:
            best_if = mean_if
            best_if_model = runnables[model_path_id]

    print('Evaluated all models.....')

    path = '/media/data/movie/dataset/tf_records'
    files = os.listdir(path)
    files = [
        fi
        for fi in files
        if fi.endswith('pkl')]
    with open(path + '/master_metics_{}.pkl'.format(str(len(files))),\
        'wb') as handle:
        pickle.dump(master_metrics, handle)
    print('Master metric file dumped.....')

    print('Best stats:')
    print('Model with lowest rep_first loss:{}, {}'.format(
        best_rf_model, best_rf))
    print('Model with lowest rep_last loss:{}, {}'.format(
        best_rl_model, best_rl))
    print('Model with lowest weighted loss:{}, {}'.format(
        best_wf_model, best_wf))
    print('Model with lowest interframe loss:{}, {}'.format(
        best_if_model, best_if))


