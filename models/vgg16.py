########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
import os


class vgg16:
    def __init__(self, imgs, end_point = 'conv4_3',
        verbose = False):
        
        self.imgs = imgs
        self.scale_image()
        self.load_weights('vgg16_weights.npz')
        self.features = self.get_features(\
            end_point, verbose = verbose)


    def get_features(self, end_point, verbose = False):
        block_name= end_point.split('_')[0].casefold()

        if block_name[:-1]=='conv':
            self.convlayers(int(block_name[-1]))
        elif block_name[:-1]=='pool':
            self.convlayers(int(block_name[-1]))
        elif block_name[:-1]=='fc':
            self.resize_image()
            self.convlayers()
            self.fc_layers(int(block_name[-1]))

        try:
            return getattr(self,end_point.casefold())
        except AttributeError:
            print('VGG layer - '+end_point+\
                ' - not present. \
                Use format convX_Y or fcX')
            return tf.constant([0.])

    def scale_image(self):
        shape = self.imgs.get_shape().as_list()

        img_min = tf.math.reduce_min(self.imgs) 
        img_max = tf.math.reduce_max(self.imgs)

        img_range = img_max - img_min

        frames = 255 * \
        (self.imgs - img_min) / img_range

        if len(shape)>4:
            frames = tf.reshape(frames,\
                [-1,shape[-3],shape[-2],shape[-1]])
        elif len(shape)==3:
            frames = tf.expand_dims(mid_frames,0)

        self.imgs = tf.image.grayscale_to_rgb(frames)

    def resize_image(self):
        self.imgs = tf.image.resize(self.imgs,(224,224))


    def convlayers(self, block=5):
        self.parameters = []

        # zero-mean input
        with tf.compat.v1.variable_scope('preprocess',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            mean = tf.constant([123.68, 116.779, 103.939]\
                ,dtype=tf.float32, shape=[1, 1, 1, 3],\
                name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.compat.v1.variable_scope('conv1_1',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv1_1_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(images, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv1_1_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.compat.v1.variable_scope('conv1_2',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv1_2_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv1_2_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool2d(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        if(block==1):
            return

        # conv2_1
        with tf.compat.v1.variable_scope('conv2_1',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv2_1_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.pool1, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv2_1_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.compat.v1.variable_scope('conv2_2',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv2_2_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv2_1, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv2_2_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool2d(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        if(block==2):
            return

        # conv3_1
        with tf.compat.v1.variable_scope('conv3_1',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv3_1_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.pool2, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv3_1_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.compat.v1.variable_scope('conv3_2',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv3_2_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv3_1, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv3_2_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.compat.v1.variable_scope('conv3_3',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv3_3_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv3_2, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv3_3_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool2d(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        if(block==3):
            return

        # conv4_1
        with tf.compat.v1.variable_scope('conv4_1',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv4_1_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.pool3, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv4_1_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.compat.v1.variable_scope('conv4_2',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv4_2_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv4_1, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer =\
                self.weights_loaded['conv4_2_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.compat.v1.variable_scope('conv4_3',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv4_3_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv4_2, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv4_3_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool2d(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        if(block==4):
            return

        # conv5_1
        with tf.compat.v1.variable_scope('conv5_1',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv5_1_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.pool4, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv5_1_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.compat.v1.variable_scope('conv5_2',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv5_2_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv5_1, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv5_2_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.compat.v1.variable_scope('conv5_3',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            kernel = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['conv5_3_W'],\
                dtype=tf.float32, trainable=False)
            conv = tf.nn.conv2d(self.conv5_2, kernel,\
                [1, 1, 1, 1], padding='SAME')
            biases = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['conv5_3_b'],\
                dtype=tf.float32, trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool2d(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')


    def fc_layers(self, block):
        # fc1
        with tf.compat.v1.variable_scope('fc1',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            shape = int(np.prod(\
                self.pool5.get_shape()[1:]))
            fc1w = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['fc6_W'],\
                dtype=tf.float32, trainable=False)
            fc1b = tf.compat.v1.get_variable(\
                name = 'biases',initializer =\
                self.weights_loaded['fc6_b'],\
                dtype=tf.float32, trainable=False)
            pool5_flat = tf.reshape(\
                self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(\
                pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        if(block==1):
            return

        # fc2
        with tf.compat.v1.variable_scope('fc2',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            fc2w = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['fc7_W'],\
                dtype=tf.float32, trainable=False)
            fc2b = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['fc7_b'],\
                dtype=tf.float32, trainable=False)
            fc2l = tf.nn.bias_add(tf.matmul(\
                self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        if(block==2):
            return

        # fc3
        with tf.compat.v1.variable_scope('fc3',\
            reuse=tf.compat.v1.AUTO_REUSE) as scope:
            fc3w = tf.compat.v1.get_variable(\
                name = 'weights',initializer=\
                self.weights_loaded['fc8_W'],\
                dtype=tf.float32, trainable=False)
            fc3b = tf.compat.v1.get_variable(\
                name = 'biases',initializer=\
                self.weights_loaded['fc8_b'],\
                dtype=tf.float32, trainable=False)
            self.fc3l = tf.nn.bias_add(tf.matmul(\
                self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file):
        path = os.getcwd() + '/models/'
        self.weights_loaded = np.load(
            path + weight_file)

def build_vgg16(imgs, end_point='conv4_3', verbose=False):
    return vgg16(imgs, end_point=end_point, verbose=verbose)

def build_vgg16(imgs, end_point = 'conv4_3',verbose = False):
    return vgg16(imgs, end_point = end_point,verbose = verbose)






# if __name__ == '__main__':
#     sess = tf.Session()
#     imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
#     vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

#     vgg.trainable = False
