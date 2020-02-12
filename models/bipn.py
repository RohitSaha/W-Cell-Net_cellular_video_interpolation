import tensorflow as tf

from models.utils.layer import linear as MLP
from models.utils.layer import conv_batchnorm_relu as CBR
from models.utils.layer import upconv_2D as UC
from models.utils.layer import maxpool as MxP
from models.utils.layer import avgpool as AvP

def build_bipn(inputs, use_batch_norm=False,
                is_training=False):

    get_shape = inputs.get_shape().as_list()

    return inputs
