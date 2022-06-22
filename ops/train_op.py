import tensorflow as tf

def get_train_op(model, loss):
    train_sover = model.params['Train_config']['solver']
    if train_sover == 'SGD':
        # SGD train
        optimizer = tf.train.MomentumOptimizer(learning_rate=model.lr.lr_var, momentum=model.lr.momentum)
    elif train_sover == 'ADAM':
        # ADAM train
        optimizer = tf.train.AdamOptimizer(learning_rate=model.lr.lr_var)
    else:
        # Default - SGD train
        optimizer = tf.train.MomentumOptimizer(learning_rate=model.lr.lr_var, momentum=model.lr.momentum)

    # Variable list for training should exclude noise_var in Attack class when attack is based on training
    var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if var.name.find('Attack') == -1]
    # train_op = optimizer.minimize(loss, var_list=var_list)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        grad_list = optimizer.compute_gradients(loss=loss, var_list=var_list)

    train_op = optimizer.apply_gradients(grads_and_vars=grad_list, name='SGD', global_step=model.global_step)

    return train_op