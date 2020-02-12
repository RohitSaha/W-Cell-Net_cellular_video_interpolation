import tensorflow as tf

from utils import layer.linear as MLP
from utils import layer.conv_batchnorm_relu as CBR
from utils import layer.upconv2D as UC
from utils import layer.maxpool as MxP
from utils import layer.avgpool as AvP

def build_bipn(inputs, use_batch_norm=False,
                is_training=False):

    get_shape = inputs.get_shape().as_list()

    return inputs
