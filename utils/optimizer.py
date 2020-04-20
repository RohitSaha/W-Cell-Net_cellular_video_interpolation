import tensorflow as tf
import numpy as np

def get_optimizer(train_loss, optim_id=1,
                    learning_rate=1e-3,
                    use_batch_norm=False,
                    var_list=[]):
    '''Defines optimizer and returns it
    Args:
        train_loss: Scalar 'Tensor' of dtype
            tf.float32
        optim_id: 'Integer' to mention the
            optimizer to be used
        learning_rate: 'Float' to specify the
            initial learning rate
        use_batch_norm: 'Bool' to mention whether
            batch norm is being used in the model
        var_list: 'List' of 'Tensors' that have to
            be optimizer. Useful for GANs
    Returns:
        Optimizer 'Tensor' that can be minimized
    '''
    if optim_id == 1:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
    elif optim_id == 2:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9,
            use_nesterov=True)

    if use_batch_norm:
        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if var_list == []:
                train_op = optimizer.minimize(
                    train_loss)
            else:
                train_op = optimizer.minimize(
                    train_loss,
                    var_list=var_list)
    else:
        if var_list == []:
            train_op = optimizer.minimize(
                train_loss)
        else:
            train_op = optimizer.minimize(
                train_loss,
                var_list=var_list)

    return train_op


def count_parameters(variables):
    '''Counts network parameters
    Args:
        variables: 'List' of 'Tensors' initialized
            in a model
    Returns:
        'Integer' specifying the number of parameters
    '''
    return np.sum(
        [np.prod(
            v.get_shape().as_list())
            for v in variables]) 


