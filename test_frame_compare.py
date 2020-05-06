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

from utils.losses import l2_loss
from utils.losses import ssim_loss

from data_pipeline.read_record import read_and_decode


def testing(info):
    

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



        # Difference between first and last frame
        l2_diff = l2_loss(
                test_fFrames, test_lFrames)
        
        dssim_diff = ssim_loss(
        		test_fFrames, test_lFrames)

        difference = {}
        difference['L2'] = []
        difference['DSSIM'] = []
        difference['largest_l2_idx']=0
        difference['largest_dssim_idx']=0

        # START TRAINING HERE
        for iteration in range(test_iters):

            # get the frame differences
            l2_val, dssim_val = sess.run(
                [l2_diff, dssim_diff])

            difference['L2'].append(l2_val)
            difference['DSSIM'].append(dssim_val)


        print('Testing complete.....')
        

    difference['largest_l2_idx'] = difference['L2'].index(
    	max(difference['L2']))

    difference['largest_dssim_idx'] = difference['DSSIM'].index(
    	max(difference['DSSIM']))

    with open(info['model_path'] + '/evaluation.pkl', 'wb') as handle:
        pickle.dump(difference, handle)

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
    # model = 'skip_conn_separate_encoder_bipn_100000_32_adam_0.001_l2_nIF-{}_startOutChannels-{}'
    info = {}

    window_size = args.window_size
    out_channels = args.out_channels

    exp_name = exp_name.format(str(window_size))
    # model = model.format(str(window_size - 2), str(out_channels))
    model = 'unet_separate_encoder_bipn_100000_32_adam_0.001_l2_nIF-3_startOutChannels-4'

    info['model_path'] = os.path.join(ROOT_DIR, exp_name, model + '/')
    info['model_name'] = 'unet'
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


