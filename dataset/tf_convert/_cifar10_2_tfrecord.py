# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts CIFAR10 data to TFRecords of TF-Example protos.

This module downloads the CIFAR10 data, uncompresses it, reads the files
that make up the CIFAR10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

#from matplotlib import pyplot as PLT
import numpy as np
import urllib
import tensorflow as tf

import tarfile


LABELS_FILENAME = 'labels.txt'

class StandardNormalDistribution():
    def __init__(self, mean=None, deviation=None):
        self.__mean = mean
        self.__deviation = deviation

    @property
    def mean(self):
        return self.__mean

    @property
    def deviation(self):
        return self.__deviation

    @mean.setter
    def mean(self, value):
        self.__mean = value

    @deviation.setter
    def deviation(self, value):
        self.__deviation = value


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
      values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
      values: A scalar of list of values.

    Returns:
      A TF-Feature.
    """
    #if not isinstance(values, (tuple, list)):
    #  values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.

    Args:
      tarball_url: The URL of a tarball file.
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
      for label in labels_to_class_names:
        class_name = labels_to_class_names[label]
        f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
      lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
      index = line.index(':')
      labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names

# The URLs where the CIFAR10 data can be downloaded.
_TRAIN_DATA_FILENAME = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
_VALID_DATA_FILENAME = ['data_batch_5']
_TEST_DATA_FILENAME = ['test_batch']

_IMAGE_SIZE = 32
_NUM_CHANNELS = 3

# Split rate for pruning validation dataset
_SPLIT_RATE = 0.1


def _extract_images(filename, num_images):
    """Extract the images into a numpy array.

    Args:
      filename: The path to an CIFAR10 images file.
      num_images: The number of images in the file.

    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
        #data = data.astype(np.float32)
    return data


def _extract_labels(filename, num_labels):
    """Extract the labels into a vector of int64 label IDs.

    Args:
      filename: The path to an CIFAR10 labels file.
      num_labels: The number of labels in the file.

    Returns:
      A numpy array of shape [number_of_labels]
    """
    print('Extracting labels from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/cifar_%s.tfrecord' % (dataset_dir, split_name)


def _add_to_tfrecord(dataset_dir, data_filename, num_images, tfrecord_writer, globalData, apply_mean):
    """Loads data from the binary CIFAR10 files and writes files to a TFRecord.

    Args:
      data_filename: The filename of the CIFAR10 images.
      labels_filename: The filename of the CIFAR10 labels.
      num_images: The number of images in the dataset.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    images = []
    labels = []

    for f in data_filename:
        batch_dir = os.path.join(dataset_dir, f)
        batch = unpickle(batch_dir)
        images.extend(batch[b'data'])
        labels.extend(batch[b'labels'])

    if apply_mean ==True:
        images = np.array((images[:] - globalData.mean) / globalData.deviation)

    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)

    for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()
        img = np.reshape(images[j],[3,32,32])

        img[0][:] = np.transpose(img[0][:])
        img[1][:] = np.transpose(img[1][:])
        img[2][:] = np.transpose(img[2][:])
        img = np.swapaxes(img, 0, 2)
        #img = np.reshape(img,[-1])
        #PLT.imshow(img)
        #PLT.show()
        img = img.tobytes()
        example = image_to_tfexample(img, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())


def _add_to_split_tfrecord(dataset_dir, data_filename, split_rate, tfrecord_writer_hdout, tfrecord_writer_rest, globalData, apply_mean):
    """Loads data from the binary CIFAR10 files and writes files to a TFRecord.

    Args:
      data_filename: The filename of the CIFAR10 images.
      labels_filename: The filename of the CIFAR10 labels.
      file_list: The indexes list of holdout images.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    images = []
    labels = []

    for f in data_filename:
        batch_dir = os.path.join(dataset_dir, f)
        batch = unpickle(batch_dir)
        images.extend(batch[b'data'])
        labels.extend(batch[b'labels'])

    num_images = len(images)

    cls_num_images = int(num_images*split_rate / 10)
    label_list = [[]] * 10

    for idx, lab in enumerate(labels):
        label_list[lab] = label_list[lab] + [idx]

    f_list = []
    rest_list = []
    for lab in range(len(label_list)):
        cls_labs = np.array(label_list[lab])
        rand_idxes = np.random.permutation(len(cls_labs))
        selet_idxes = rand_idxes[:cls_num_images]
        rest_idxes = rand_idxes[cls_num_images:]
        f_list.extend(cls_labs[selet_idxes])
        rest_list.extend(cls_labs[rest_idxes])

    # Shuffle f_list for holdout & rest
    np.random.shuffle(f_list)
    np.random.shuffle(rest_list)

    if apply_mean == True:
        images = np.array((images[:] - globalData.mean) / globalData.deviation)

    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)

    # Generate split-out dataset for Validation in pruning
    print(">> Holdout images")
    for i, idx in enumerate(f_list):
        sys.stdout.write('\r>> Converting holdout image %d/%d' % (i + 1, int(num_images*split_rate)))
        sys.stdout.flush()
        img = np.reshape(images[idx], [3, 32, 32])

        img[0][:] = np.transpose(img[0][:])
        img[1][:] = np.transpose(img[1][:])
        img[2][:] = np.transpose(img[2][:])
        img = np.swapaxes(img, 0, 2)
        # img = np.reshape(img,[-1])
        # PLT.imshow(img)
        # PLT.show()
        img = img.tobytes()
        example = image_to_tfexample(img, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[idx])
        tfrecord_writer_hdout.write(example.SerializeToString())

    # Generate rest dataset for Training in pruning
    print("\n>> Rest images")
    for j, idx in enumerate(rest_list):
        sys.stdout.write('\r>> Converting rest image %d/%d' % (j + 1, int(num_images*(1-split_rate))))
        sys.stdout.flush()
        img = np.reshape(images[idx], [3, 32, 32])

        img[0][:] = np.transpose(img[0][:])
        img[1][:] = np.transpose(img[1][:])
        img[2][:] = np.transpose(img[2][:])
        img = np.swapaxes(img, 0, 2)
        # img = np.reshape(img,[-1])
        # PLT.imshow(img)
        # PLT.show()
        img = img.tobytes()
        example = image_to_tfexample(img, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[idx])
        tfrecord_writer_rest.write(example.SerializeToString())


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if not tf.gfile.Exists(dataset_dir+'/cifar-10-batches-py'):
        download_and_uncompress_tarball('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', dataset_dir)

    train_filename = _get_output_filename(dataset_dir, 'train')
    valid_filename = _get_output_filename(dataset_dir, 'valid')
    train_holdout_filename = _get_output_filename(dataset_dir, 'train_5k')
    train_rest_filename = _get_output_filename(dataset_dir, 'train_45k')
    test_filename = _get_output_filename(dataset_dir, 'test')

    dataset_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')

    if tf.gfile.Exists(train_filename) and tf.gfile.Exists(valid_filename) and tf.gfile.Exists(test_filename) and \
            tf.gfile.Exists(train_holdout_filename) and tf.gfile.Exists(train_rest_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    apply_mean = False
    globalData = StandardNormalDistribution(mean=0,deviation=0)

    # Training Data:
    print ('\nConverting Training-dataset:')
    with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:
        #data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
        #labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
        _add_to_tfrecord(dataset_dir, _TRAIN_DATA_FILENAME, 50000, tfrecord_writer, globalData, apply_mean)

    # Split Train Dataset for Pruning:
    print('\nConverting Split-Training-dataset:')
    tfrecord_writer_hdout = tf.python_io.TFRecordWriter(train_holdout_filename)
    tfrecord_writer_rest = tf.python_io.TFRecordWriter(train_rest_filename)
    # data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
    # labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
    _add_to_split_tfrecord(dataset_dir, _TRAIN_DATA_FILENAME, _SPLIT_RATE, tfrecord_writer_hdout, tfrecord_writer_rest, globalData, apply_mean)

    # Validation Data:
    print('\nConverting Valid-dataset:')
    with tf.python_io.TFRecordWriter(valid_filename) as tfrecord_writer:
        _add_to_tfrecord(dataset_dir, _VALID_DATA_FILENAME, 10000, tfrecord_writer, globalData, apply_mean)

    # Testing data:
    print('\nConverting Test-dataset:')
    with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
        _add_to_tfrecord(dataset_dir, _TEST_DATA_FILENAME, 10000, tfrecord_writer, globalData, apply_mean)

    print('\nFinished converting the CIFAR-10 dataset!')


if __name__ == '__main__':
    ds_dir = './data/CIFAR10'
    run(ds_dir)

