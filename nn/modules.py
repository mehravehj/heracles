import tensorflow as tf
from nn import layer


# First ResNet Block for Cifar

def _residual_block_first_cifar(x, co, k, stride, params, prune_reg=''):
    shortcut_1 = layer.get_max_pool(x, 1, 2, pad='SAME')
    shortcut_2 = tf.zeros(shape=[x.get_shape().as_list()[0],
                                 x.get_shape().as_list()[1] / 2,
                                 x.get_shape().as_list()[2] / 2,
                                 x.get_shape().as_list()[3]])
    shortcut = tf.concat([shortcut_1, shortcut_2], axis=3)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, shortcut)
    shortcut = layer.get_bn_layer(shortcut)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, shortcut)

    _conv_1 = layer.get_conv_layer(x, co=co, k=k, stride=2, params=params, name='_1', prune_reg=prune_reg, mask_name='_1')
    with tf.variable_scope('bn_1'):
        _conv_1 = layer.get_bn_layer(_conv_1)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_1 = layer.get_relu(_conv_1, name='Relu_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_2 = layer.get_conv_layer(_conv_1, co=co, k=k, stride=stride, params=params, name='_2', prune_reg=prune_reg, mask_name='_2')
    with tf.variable_scope('bn_2'):
        _conv_2 = layer.get_bn_layer(_conv_2)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_2)

    block_out = _conv_2 + shortcut
    block_out = layer.get_relu(block_out, name='Relu_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, block_out)

    return block_out


# Preact ResNet Block

def _residual_block_first_preact(x, co, k, stride, params, prune_reg=''):

    _bn_1 = layer.get_bn_layer(x, name='bn_1')
    if _bn_1 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _bn_1)

    _act_1 = layer.get_relu(_bn_1, name='Relu_1')
    if _act_1 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _act_1)

    shortcut = layer.get_conv_layer(_act_1, co=co, k=1, stride=stride, params=params, name='_shortcut',
                                    prune_reg=prune_reg,
                                    mask_name='_shortcut', pad='VALID')

    _conv_1 = layer.get_conv_layer(_act_1, co=co, k=k, stride=stride, params=params, name='_1', prune_reg=prune_reg, mask_name='_1')

    _bn_2 = layer.get_bn_layer(_conv_1, name='bn_2')
    if _bn_2 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _bn_2)

    _act_2 = layer.get_relu(_bn_2, name='Relu_2')
    if _act_2 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _act_2)

    _conv_2 = layer.get_conv_layer(_act_2, co=co, k=k, stride=1, params=params, name='_2', prune_reg=prune_reg,
                                   mask_name='_2')

    block_out = _conv_2 + shortcut
    if block_out not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, block_out)

    return block_out


def _residual_block_preact(x, co, k, stride, params, prune_reg=''):
    shortcut = x

    _bn_1 = layer.get_bn_layer(x, name='bn_1')
    if _bn_1 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _bn_1)

    _act_1 = layer.get_relu(_bn_1, name='Relu_1')
    if _act_1 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _bn_1)

    _conv_1 = layer.get_conv_layer(_act_1, co=co, k=k, stride=1, params=params, name='_1', prune_reg=prune_reg, mask_name='_1')

    _bn_2 = layer.get_bn_layer(_conv_1, name='bn_2')
    if _bn_2 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _bn_2)

    _act_2 = layer.get_relu(_bn_2, name='Relu_2')
    if _act_2 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _act_2)

    _conv_2 = layer.get_conv_layer(_act_2, co=co, k=k, stride=1, params=params, name='_2', prune_reg=prune_reg, mask_name='_2')

    block_out = _conv_2 + shortcut
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, block_out)

    return block_out


# Wide ResNet First Block at Beginning

def _residual_block_first_wide(x, co, k, stride, params, prune_reg=''):
    shortcut = layer.get_conv_layer(x, co=co, k=1, stride=1, params=params, name='_shortcut', prune_reg=prune_reg, mask_name='_shortcut')

    _conv_1 = layer.get_bn_layer(x, name='bn_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_1 = layer.get_relu(_conv_1, name='Relu_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_1 = layer.get_conv_layer(_conv_1, co=co, k=k, stride=1, params=params, name='_1', prune_reg=prune_reg,
                                   mask_name='_1')

    _conv_2 = layer.get_bn_layer(_conv_1, name='bn_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_2)

    _conv_2 = layer.get_relu(_conv_1, name='Relu_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_2)

    _conv_2 = layer.get_conv_layer(_conv_1, co=co, k=k, stride=stride, params=params, name='_2', prune_reg=prune_reg,
                                   mask_name='_2')

    block_out = _conv_2 + shortcut
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, block_out)

    return block_out


# Normal ResNet Block

def _residual_block_first(x, co, k, stride, params, prune_reg=''):
    # Conv_shortcut
    shortcut = layer.get_conv_layer(x, co=co, k=1, stride=2, params=params, name='_shortcut', prune_reg=prune_reg,
                                    mask_name='_shortcut', pad='VALID')

    # Conv1
    _conv_1 = layer.get_conv_layer(x, co=co, k=k, stride=2, params=params, name='_1', prune_reg=prune_reg,
                                   mask_name='_1')

    _conv_1 = layer.get_bn_layer(_conv_1, name='bn_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_1 = layer.get_relu(_conv_1, name='Relu_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    # Conv2
    _conv_2 = layer.get_conv_layer(_conv_1, co=co, k=k, stride=stride, params=params, name='_2', prune_reg=prune_reg,
                                   mask_name='_2')

    _conv_2 = layer.get_bn_layer(_conv_2, name='bn_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_2)

    shortcut_bn = layer.get_bn_layer(shortcut, name='bn_shortcut')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, shortcut_bn)

    # Add shortcut
    block_out = _conv_2 + shortcut_bn
    block_out = layer.get_relu(block_out, name='Relu_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, block_out)

    return block_out


def _residual_block(x, co, k, stride, params, prune_reg=''):
    shortcut = x
    _conv_1 = layer.get_conv_layer(x, co=co, k=k, stride=stride, params=params, name='_1', prune_reg=prune_reg, mask_name='_1')

    _conv_1 = layer.get_bn_layer(_conv_1, name='bn_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_1 = layer.get_relu(_conv_1, name='Relu_1')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_1)

    _conv_2 = layer.get_conv_layer(_conv_1, co=co, k=k, stride=stride, params=params, name='_2', prune_reg=prune_reg, mask_name='_2')

    _conv_2 = layer.get_bn_layer(_conv_2, name='bn_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _conv_2)

    block_out = _conv_2 + shortcut
    block_out = layer.get_relu(block_out, name='Relu_2')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, block_out)

    return block_out


def _residual_block_bottleneck(x, co, k, stride, params, layers=2, prune_reg=''):

    if not isinstance(k, list):
        k = [k for _ in range(layers)]

    if not isinstance(stride, list):
        stride = [stride for _ in range(layers)]

    if not isinstance(co, list):
        co = [co for _ in range(layers)]

    # Conv_shortcut
    if co[-1] != x.shape[-1]:
        shortcut_stride = stride[-2]  # In resnet, stride for shortcut is always same as -2th conv_layer
        shortcut = layer.get_conv_layer(x, co=co[-1], k=1, stride=shortcut_stride, params=params, name='_shortcut',
                                        prune_reg=prune_reg, mask_name='_shortcut', pad='VALID')
        shortcut = layer.get_bn_layer(shortcut, name='bn_shortcut')
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, shortcut)
    else:
        shortcut = x

    input_var = x
    for l in range(layers):
        name_end = '_' + str(l+1)

        if k[l] == 1:
            pad = 'VALID'
        else:
            pad = 'SAME'

        _l_conv = layer.get_conv_layer(input_var, co=co[l], k=k[l], stride=stride[l], params=params, pad=pad,
                                       name=name_end, prune_reg=prune_reg, mask_name=name_end)

        _l_bn = layer.get_bn_layer(_l_conv, name='bn'+name_end)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _l_bn)

        if l+1 == layers:
            _l_bn = shortcut + _l_bn

        _l_relu = layer.get_relu(_l_bn, name='Relu' + name_end)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _l_relu)

        input_var = _l_relu

    block_out = input_var

    return block_out
