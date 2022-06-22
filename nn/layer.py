import tensorflow as tf


def get_conv_layer(x, co, k, stride, params, bias=False, pad='SAME', prune_reg='', name='', mask_name='', trainable=True):
    if params['weight']['init']['type'] == 'rand':
        init = tf.random_normal_initializer(stddev=params['weight']['init']['std'])
    elif params['weight']['init']['type'] == 'const':
        init = tf.constant_initializer(params['weight']['init']['val'])
    elif params['weight']['init']['type'] == 'xavier':
        init = tf.contrib.layers.xavier_initializer()
    elif params['weight']['init']['type'] == 'he':
        init = tf.variance_scaling_initializer(scale=2.0, mode='fan_out')
    # elif params['weight']['init']['type'] == 'pre_trained':
    #     init = tf.constant_initializer(value=data[params['name']]['weights'])
    else:
        init = tf.random_normal_initializer(stddev=params['weight']['init']['std'])

    x_shape = x.get_shape()
    ci = x_shape[3]
    shape = [k, k, ci, co]
    w_var = tf.get_variable(name='weight' + name, shape=shape, initializer=init, trainable=trainable,
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

    # Switch to pytorch padding mode when stride != 1:
    if stride % 2 == 0 and pad != 'VALID':
        if isinstance(pad, int):
            x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='CONSTANT')
        else:
            x = tf.pad(x, paddings=[[0,0], [1,1], [1,1], [0,0]], mode='CONSTANT')
        pad = 'VALID'

    if prune_reg != '':
        _p_mask = get_prune_mask(w_var, prune_reg, mask_name)
        w_var *= _p_mask

    if not bias:
        _layer_out = tf.nn.conv2d(x, w_var, [1, stride, stride, 1], padding=pad, name='f_map' + name)
    else:
        if params['bias']['init']['type'] == 'rand':
            b_init = tf.random_normal_initializer(stddev=params['bias']['init']['std'])
        elif params['bias']['init']['type'] == 'const':
            b_init = tf.constant_initializer(params['bias']['init']['val'])
        elif params['bias']['init']['type'] == 'xavier':
            b_init = tf.contrib.layers.xavier_initializer()
        else:
            b_init = tf.random_normal_initializer(stddev=params['bias']['init']['std'])

        _bias = tf.get_variable(name="bias" + name, shape=[co], initializer=b_init, trainable=trainable)
        if _bias not in tf.get_collection(tf.GraphKeys.BIASES):
            tf.add_to_collection(tf.GraphKeys.BIASES, _bias)

        _conv = tf.nn.conv2d(x, w_var, [1, stride, stride, 1], padding=pad)
        _layer_out = tf.nn.bias_add(_conv, bias=_bias, name='f_map' + name)

    if _layer_out not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _layer_out)

    return _layer_out


def get_fc_layer(x, co, params, prune_reg='', trainable=True, name=''):
    if params['weight']['init']['type'] == 'rand':
        init = tf.random_normal_initializer(stddev=params['weight']['init']['std'])
    elif params['weight']['init']['type'] == 'const':
        init = tf.constant_initializer(params['weight']['init']['val'])
    elif params['weight']['init']['type'] == 'xavier':
        init = tf.contrib.layers.xavier_initializer()
    elif params['weight']['init']['type'] == 'he':
        init = tf.initializers.he_normal()
    # elif params['weight']['init']['type'] == 'pre_trained':
    #     init = tf.constant_initializer(value=data[params['name']]['weights'])
    else:
        init = None

    x_shape = x.get_shape()
    ci = x_shape[-1]
    shape = [ci, co]
    w_var = tf.get_variable(name='weight' + name, shape=shape, initializer=init, trainable=trainable,
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

    if prune_reg != '':
        _p_mask = get_prune_mask(w_var, prune_reg)
        w_var *= _p_mask

    b_var = tf.get_variable(name="bias" + name, shape=[co], initializer=init, trainable=trainable,
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])
    _fc = tf.matmul(x, w_var)
    _fc = tf.nn.bias_add(_fc, bias=b_var, name='f_map')
    if _fc not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _fc)

    return _fc


def get_bn_layer(x, momentum=0.9, name='bn'):
    _BN_MOMENTUM = momentum
    _BN_EPSILON = 1e-5
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
    _bn = tf.layers.batch_normalization(x,
                                        momentum=_BN_MOMENTUM,
                                        epsilon=_BN_EPSILON,
                                        training=tf.cast(is_training, tf.bool),
                                        center=True,
                                        scale=True,
                                        reuse=False,
                                        name=name)

    # with tf.variable_scope('bn'):
    #     _bn = tf.keras.layers.BatchNormalization(epsilon=1e-5)
    #     _bn.build(x.shape)
    #     _bn_output = _bn(x, training=tf.cast(is_training, tf.bool))

    return _bn


def get_relu(x, name=None):
    _relu_act = tf.nn.relu(x) if name is None else tf.nn.relu(x, name)

    if _relu_act not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _relu_act)

    return _relu_act


def get_flatten(x, name=''):
    _flatten = tf.layers.flatten(x, name)

    if _flatten not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _flatten)

    return _flatten


def get_max_pool(x, k, stride, pad='SAME'):
    if stride % 2 == 0 and pad != 'VALID':
        if isinstance(pad, int):
            x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='CONSTANT')
        else:
            x = tf.pad(x, paddings=[[0,0], [1,1], [1,1], [0,0]], mode='CONSTANT')
        pad = 'VALID'

    _max_pool = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=pad, name='max_pool')
    if _max_pool not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _max_pool)

    return _max_pool


def get_avg_pool(x, k, stride, pad='SAME'):
    _avg_pool = tf.nn.avg_pool2d(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=pad, name='avg_pool')
    if _avg_pool not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _avg_pool)

    return _avg_pool


def get_dropout(x, rate):
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
    _dropout = tf.layers.dropout(x, rate, training=tf.cast(is_training, tf.bool))

    return _dropout


def get_prune_mask(x, prune_reg='channel_prune', name=''):

    l_shape = x.get_shape().as_list()
    dims = len(l_shape)
    [ch, fi] = l_shape[-2:]

    on_fc = False if dims == 4 else True
    k = 1 if on_fc else l_shape[0]

    if prune_reg == 'channel':
        mask_shape = [ch,1] if on_fc else [1,1,ch,1]
    elif prune_reg == 'filter':
        mask_shape = [1,fi] if on_fc else [1,1,1,fi]
    elif prune_reg == 'kernel':
        mask_shape = [ch,fi] if on_fc else [1,1,ch,fi]
    elif prune_reg == 'weight':
        mask_shape = [ch,fi] if on_fc else [k,k,ch,fi]
    else:
        raise NameError('prune_reg is wrongly defined, please recheck its grammar.')

    score_var = tf.get_variable(name='p_mask' + name,
                                shape=mask_shape,
                                initializer=tf.ones_initializer,
                                trainable=True,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])

    rate_var = tf.get_variable(name='p_rate' + name,
                               shape=[],
                               initializer=tf.ones_initializer,
                               trainable=False,
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])

    b_mask = get_binary_mask(score_var, rate_var, mask_shape)

    # preserved_ch = tf.cast(mask_length * rate_var, tf.int32)
    #
    # importance = tf.reshape(score_var, shape=[mask_length])
    #
    # max_values, preserve_idx = tf.nn.top_k(input=importance, k=preserved_ch)
    # threshold = tf.reduce_min(max_values)  # In case that the dim equal to threshold is pruned
    # mask_round = tf.cast(tf.greater_equal(importance, threshold), dtype=tf.float32)
    # mask_flat = importance + tf.stop_gradient(mask_round - importance)  # Stop backpropagation on rounded mask
    # mask_flat = tf.cast(mask_flat, tf.float32)
    #
    # mask_out = tf.reshape(mask_flat, shape=mask_shape, name='p_mask_out')

    mask_out = score_var + tf.stop_gradient(b_mask - score_var)

    return mask_out


def get_binary_mask(score, rate, mask_shape):
    mask_len = tf.cast(tf.reduce_prod(mask_shape), dtype=tf.float32)
    preserved_ch = tf.cast(mask_len * rate, tf.int32)

    importance = tf.reshape(score, shape=[mask_len])

    max_values, preserve_idx = tf.nn.top_k(input=importance, k=preserved_ch)
    threshold = tf.reduce_min(max_values)  # In case that the dim equal to threshold is pruned
    mask_round = tf.cast(tf.greater_equal(importance, threshold), dtype=tf.float32)
    mask_flat = tf.cast(mask_round, tf.float32)

    mask_out = tf.reshape(mask_flat, shape=mask_shape, name='p_mask_out')

    return mask_out