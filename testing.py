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
from models.utils.optimizer import count_parameters
from models.utils.losses import huber_loss
from models.utils.losses import l2_loss
from models.utils.visualizer import visualize_frames
from models.utils.metrics import metric_repeat_fframe
from models.utils.metrics import metric_repeat_lframe
from models.utils.metrics import metric_weighted_frame
from models.utils.metrics import metric_interpolated_frame

from models import skip_separate_encoder_bipn
from models import skip_unet_separate_encoder_bipn
from models import vgg16



def get_files():
    ROOT_DIR = '/media/data/movie/dataset/tf_records/'
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
    try:
        loss_index = details.index('l2')
    except:
        loss_index = details.index('l1')
    
    loss = details[loss_index]
    nIF = int(details[loss_index + 1][-1])
    out_channels = int(details[loss_index + 2].split(
        '-')[-1])
    attention = details[loss_index + 3]
    additional_info = details[-1]

    return model_name, loss, nIF, out_channels,\
        attention, additional_info
            

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
    model_name, loss, nIF, out_channels, attention,\
        additional_info = get_model_details(
            model_path)

    # get #test_samples based on experiment
    if n_IF == 3: test_samples = 18300
    elif n_IF == 4: test_samples = 18296
    elif n_IF == 5: test_samples = 18265
    elif n_IF == 6: test_samples = 18235
    elif n_IF == 7: test_samples = 18204
    test_iters = test_samples // args.batch_size

    # get attention
    if attention == 'spatialAttention':
        use_attention = 1
        spatial_attention = 1
    elif attention == 'channelAttention':
        use_attention = 1
        spatial_attention = 0

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
                    starting_out_channels=out_channels,
                    use_attention=use_attention,
                    spatial_attention=spatial_attention,
                    is_verbose=True)

            elif model_name == 'unet':
                test_rec_iFrames = skip_unet_separate_encoder_bipn.build_bipn(
                    test_fFrames,
                    test_lFrames,
                    use_batch_norm=True,
                    is_training=False,
                    n_IF=n_IF,
                    starting_out_channels=out_channels,
                    use_attention=use_attention,
                    spatial_attention=spatial_attention,
                    is_verbose=True)

        print('Global parameters:{}'.format(
            count_parameters(tf.global_variables())))
        print('Learnable model parameters:{}'.format(
            count_parameters(tf.trainable_variables())))

        # DEFINE LOSS 
        if loss == 'l2':
            test_loss = l2_loss(
                test_iFrames, test_rec_iFrames)
        elif loss == 'l1':
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

            # get frames and metrics
            start_frames, end_frames, mid_frames, rec_mid_frames,\
            repeat_first, repeat_last, weighted, true_metric = sess.run(
                [test_fFrames, test_lFrames, test_iFrames, test_rec_iFrames,\
                repeat_fFrame, repeat_lFrame, weighted_frame,\
                    inter_frame])

            metrics['repeat_first'].append(repeat_first)
            metrics['repeat_last'].append(repeat_last)
            metrics['weighted_frames'].append(weighted)
            metrics['inter_frames'].append(true_metric)

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
        with open(model_path + '/evaluation.pkl', 'wb')\
            as handle:
            pickle.dump(metrics, handle)
        print('Pickle file dumped.....')

    return metrics


def control(args):

    # get runnable files
    runnables = get_files()
    master_metrics = {}
    best_rf = float('inf')
    best_rl = float('inf')
    best_wf = float('inf')
    best_if = float('inf')

    for model_path_id in range(len(runnables)):
        metrics = testing(
            runnable[model_path_id],
            args)        

        rep_first = metrics['repeat_first']
        rep_last = metrics['repeat_last']
        weight_frames = metrics['weighted_frames']
        inter_frames = metrics['inter_frames']
    
        mean_rf = sum(rep_first) / len(rep_first)
        mean_rl = sum(rep_last) / len(rep_last)
        mean_wf = sum(weight_frames) / len(weight_frames)
        mean_if = sum(inter_frames) / len(inter_frames)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='To mention the number of samples in a batch')

    parser.add_argument(
        '--debug',
        type=int,
        default=1,
        help='Specifies whether to run the script in DEBUG mode')

    args = parser.parse_args()

    control(args)

