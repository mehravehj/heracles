import tensorflow as tf


def regularization(collection, params=None):
    weight_collection = tf.get_collection(tf.GraphKeys.WEIGHTS)
    regularization = params['Train_config']['regularization']
    if regularization:
        reg_type = regularization
        reg_weight = params['Train_config']['reg_weight']
        if reg_type == "l2":
            for var in weight_collection:
                reg_loss = tf.multiply(tf.nn.l2_loss(var), reg_weight, name='weight_loss')
                if reg_loss not in tf.get_collection(collection):
                    tf.add_to_collection(collection, reg_loss)
        # elif reg_type == "l1":
        #     for var in weight_collection:
        #         reg_loss = tf.multiply(tf.nn.l1_loss(var), reg_weight, name='weight_loss')
        #         if reg_loss not in tf.get_collection(collection):
        #             tf.add_to_collection(collection, reg_loss)


def loss_op(logits, labels, params):

    batch_size = params['Meta']['batchsize']

    logits_shape = logits.get_shape().as_list()
    # if logits_shape[0] > batch_size:
    #     logits = logits[:batch_size]
    # else:
    #     pass

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    pred_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.summary.scalar('Pred_loss', pred_loss)

    collection = tf.GraphKeys.REGULARIZATION_LOSSES

    if params['Train_config']['regularization']:
        regularization(collection, params=params)

    if tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add_n([pred_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    else:
        reg_loss = tf.constant(0.0)
        total_loss = pred_loss

    tf.summary.scalar('total_loss', total_loss)

    return total_loss, pred_loss, reg_loss


# def kl_div_loss(logits_t, logits):
#
#     soft_aa = tf.nn.softmax(logits)
#     soft_bng = tf.nn.softmax(logits_t)
#
#     kl_div = tf.keras.losses.KLDivergence()
#     kl_loss = kl_div(soft_bng, soft_aa)
#
#     return kl_loss
#
#
# def trades_loss(logits, logits_bng, labels, params):
#
#     batch_size = params['Meta']['batchsize']
#
#     logits_shape = logits.get_shape().as_list()
#     if logits_shape[0] > batch_size:
#         logits = logits[:batch_size]
#         logits_bng = logits[:batch_size]
#     else:
#         pass
#
#     beta = params['Attack_config']['trades_beta']
#
#     # Compute xent loss on adv. examples
#     xent_loss, _ = loss_op(logits, labels, params)
#
#     # Compute KL-Divergence Loss
#     kl_loss = kl_div_loss(logits_bng, logits)
#
#     # Compute xent loss on bng. examples
#     bng_loss, bng_reg_loss = loss_op(logits_bng, labels, params)
#
#     trades_loss = bng_loss + beta * kl_loss
#     trades_reg_loss = bng_reg_loss  # Actually useless for training
#
#     return kl_loss, xent_loss, trades_loss, trades_reg_loss
