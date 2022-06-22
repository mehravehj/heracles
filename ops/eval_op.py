import tensorflow as tf


def get_eval_op(logits, labels, batch_size):
    logits_shape = logits.get_shape().as_list()
    if logits_shape[0] > batch_size:
        logits = logits[:batch_size]

    y = tf.nn.softmax(logits)
    preds = tf.argmax(y, axis=-1)
    y = tf.cast(y, dtype=tf.float32)
    top_1 = tf.nn.in_top_k(y, labels, 1)
    top_5 = tf.nn.in_top_k(y, labels, 5)

    acc_1 = tf.reduce_mean(tf.cast(top_1, dtype=tf.float32)) * 100.0
    acc_5 = tf.reduce_mean(tf.cast(top_5, dtype=tf.float32)) * 100.0

    return acc_1, acc_5, preds, y

