import tensorflow as tf

"""@package docstring
Functions to access and handle tfrecords.
"""

def tfrecord_read_batch(img, label, batchsize, shuffle = True, capacity = 10000, min_after_dequeue = 10000):
    """Read an entire batch from a tfrecord-file.
    Call tfrecord_read_and_decode_single_example() to get an image and a label as constructor.
    """
    if isinstance(label, list):
        input = [img] + label
    else:
        input = [img, label]

    if shuffle == True:
        output = tf.train.shuffle_batch(input, num_threads=24, batch_size=batchsize, capacity=capacity, min_after_dequeue=min_after_dequeue)
    else:
        output = tf.train.batch(input, batch_size=batchsize, capacity=capacity, num_threads=4)

    if isinstance(label, list):
        return output[0], output[1:]
    else:
        images_batch, labels_batch = output

    return images_batch, labels_batch


def tfrecord_read_and_decode_single_example(filenames, shape, format='uint8'):
    """Get one image and label from tfrecords.
    Also used as constructor to read a batch. Use image and label for this
    """
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue', num_epochs=None)
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    if format == 'classification':
        tfrecord_features = _parse_classification_feature(tfrecord_serialized)
    elif format == 'object_detection':
        tfrecord_features = _parse_object_detection_feature(tfrecord_serialized)
    elif format == 'segmentation':
        tfrecord_features = _parse_segmentation_featue(tfrecord_serialized)
    else:
        tfrecord_features = _parse_classification_feature(tfrecord_serialized)

    image = tf.decode_raw(tfrecord_features['image/encoded'], tf.uint8)

    # image_shape = [tfrecord_features['image/height'], tfrecord_features['image/width'], shape[-1]]

    image = tf.reshape(image, shape=[shape[0]*shape[1]*shape[2]])

    if format == 'classification':
        label = tfrecord_features['image/class/label']
    elif format == 'object_detection':
        label1 = tfrecord_features['image/object/class/label']
        xmin = tfrecord_features['image/object/bbox/xmin']
        xmax = tfrecord_features['image/object/bbox/xmax']
        ymin = tfrecord_features['image/object/bbox/ymin']
        ymax = tfrecord_features['image/object/bbox/ymax']
        total_obj = tf.cast(tfrecord_features['image/tot_obj'], tf.int32)
        label = [total_obj, label1, ymin, ymax, xmax, xmax, total_obj]
    elif format == 'segmentation':
        #label = tfrecord_features['image/class/label']
        label = tf.decode_raw(tfrecord_features['image/class/label'], tf.uint8)
        label = tf.reshape(label, shape=[shape[1], shape[0]]) #1024x2048
    #format = tf.decode_raw(tfrecord_features['image/format'], tf.uint8)
    #height = tfrecord_features['image/height']
    #width = tfrecord_features['image/width']

    return image, label

def _parse_object_detection_feature(serialized):
    return tf.parse_single_example(serialized, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/tot_obj': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32)}, name='features')

def _parse_classification_feature(serialized):
    return tf.parse_single_example(serialized, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64)}, name='features')

def _parse_segmentation_featue(serialized):
    return tf.parse_single_example(serialized, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64)}, name='features')


# def tfrecord_read_HAR_sample(filenames):
#     tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue', num_epochs=None)
#     reader = tf.TFRecordReader()
#     _, tfrecord_serialized = reader.read(tfrecord_file_queue)
#     tfrecord_features = tf.parse_single_example(tfrecord_serialized, features={
#         'feature': tf.FixedLenFeature([120], tf.float32),
#         'label': tf.FixedLenFeature([6], tf.float32)})
#     image = tfrecord_features['feature']
#     label = tfrecord_features['label']
#     #image = tf.reshape(image, shape=[120])
#     #label = tf.reshape(label, shape=[6])
#     return image, label