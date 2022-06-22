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
r"""Downloads and converts ImageNet data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
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
import time
import lmdb

import numpy as np
import urllib
import os

import torch
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
from PIL import Image
import math
import lmdb

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

_TRAIN_DATA_FILENAME = 'train'
_VALID_DATA_FILENAME = 'val'
_TEST_DATA_FILENAME = 'test'
_LABELS_FILENAME = 'train_label.txt'
_VALID_LABELS_FILENAME = 'valid_label.txt'
_SPLIT_LABEL_FILENAME = 'train_split_label.txt'

_IMAGE_SIZE = 224
_NUM_CLASSES = 1000
_NUM_CHANNELS = 3

_IMG_OFFSET = 196620


def tf_gpu_config(parameter):
    config = tf.ConfigProto()
    config.log_device_placement = '0'
    # config.gpu_options.visible_device_list = parameter['Gpu_config']['cuda_visible_devices']
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

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
    filename: The path to an MNIST labels file.
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

def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
        dataset_dir: The directory where the temporary files are stored.
        split_name: The name of the train/test split.

    Returns:
        An absolute file path.
    """
    return '%s/ilsvrc12_%s.tfrecord' % (dataset_dir, split_name)


# def _clean_up_temporary_files(dataset_dir):
#   """Removes temporary files used to create the dataset.
#
#   Args:
#     dataset_dir: The directory where the temporary files are stored.
#   """
#   for filename in [_TRAIN_DATA_FILENAME,
#                    _VALID_DATA_FILENAME,
#                    _TEST_DATA_FILENAME
#                    _LABELS_FILENAME]:
#     filepath = os.path.join(dataset_dir, filename)
#     tf.gfile.Remove(filepath)


def _is_cmyk(img_fname):
    black_list = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                  'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                  'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                  'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                  'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                  'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                  'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                  'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                  'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                  'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                  'n07583066_647.JPEG', 'n13037406_4650.JPEG',
                  'ILSVRC2012_val_00019877.JPEG']

    return img_fname.split('/')[-1] in black_list


def _is_png(img_fname):
    return 'n02105855_2933.JPEG' == img_fname.split('/')[-1]


def _convert_2_rbg(img, img_fname):
    # if _is_png(img_fname) or _is_cmyk(img_fname):
    #     img = img.convert('RGB')
    # elif img.mode == 'L':
    #     img = img.convert('RGB')

    # if img.format != 'JPEG':
    #     print(img_fname, ' is not a JPEG file')
    #     img = img.convert('RGB')
    # elif img.mode != 'RGB':
    #     print(img_fname, ' has no RGB colospace but ', img.mode)
    #     if img.mode == 'CMYK':
    #         print('CMYK')
    #     img = img.convert('RGB')
    # elif img.layers != 3:
    #     print(img_fname, ' has merely %d channels' %img.layers)
    #     img = img.convert('RGB')
    # else:
    #     invalid = False

    if img.format != 'JPEG' or img.mode != 'RGB' or img.layers != 3:
        img = img.convert('RGB')

    return img


def random_size(image, target_size=None):
    height, width = image.size
    if height < width:
        size_ratio = target_size / height
        resize_shape = (_IMAGE_SIZE, int(width * size_ratio))
    elif height < width:
        size_ratio = target_size / width
        resize_shape = (int(height * size_ratio), _IMAGE_SIZE)
    else:
        resize_shape = (_IMAGE_SIZE, _IMAGE_SIZE)
    return image.resize(resize_shape, Image.BILINEAR)


def _add_to_tfrecord(data_filename, num_images, tfrecord_writer, as_trainset=False, as_splittrain=False):

    pos = 0

    if as_splittrain:
        class_n = np.ceil(num_images / _NUM_CLASSES)
        class_n = int(class_n)

    f_list = os.listdir(data_filename)
    f_list.sort()

    start_time = time.time()
    for i, f_n in enumerate(f_list):
        label = int(i)  # original label is 1-1000, however in real labels are in [0, 999]

        f_path = os.path.join(data_filename, f_n)
        img_list = os.listdir(f_path)

        if as_splittrain:
            np.random.shuffle(img_list)
            img_list = img_list[:class_n]
        else:
            img_list.sort()

        for im_n in img_list:

            img_file = os.path.join(f_path, im_n)

            try:
                img = Image.open(img_file)
                img.verify()
                img.close()
            except:
                raise NameError('Found trouble file:' + data_filename)

            img = Image.open(img_file)
            img = _convert_2_rbg(img, img_file)

            if as_trainset:
                transformer = transforms.Compose([
                    transforms.RandomResizedCrop(_IMAGE_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                # img = random_size(img, _IMAGE_SIZE)
            else:
                transformer = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(_IMAGE_SIZE),
                    transforms.ToTensor(),
                ])
                # img = img.resize([_IMAGE_SIZE, _IMAGE_SIZE], resample=Image.BILINEAR)

            img = transformer(img).numpy()
            img = np.rollaxis(img, 0, 3)

            img = np.asarray(img*255).astype(np.uint8)

            # img = np.reshape(img, [_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])
            # img = np.reshape(img, [_IMAGE_SIZE * _IMAGE_SIZE * _NUM_CHANNELS])
            img = img.tobytes()

            example = image_to_tfexample(img, 'jpeg'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, label)
            tfrecord_writer.write(example.SerializeToString())
            pos += 1
            sys.stdout.write(
                '\r>> Converting image %d/%d   --   Duration: %.0f s   --   %.2f percent Completed' % (
                pos, num_images, time.time() - start_time, 100 * (pos / num_images)))
            sys.stdout.flush()

            if pos / num_images == 1.0:
                break

def run():
    dataset_dir = './dataset/ImageNet'
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    test_filename = _get_output_filename(dataset_dir, 'test')
    training_filename = _get_output_filename(dataset_dir, 'train')
    trainsplit_filename = _get_output_filename(dataset_dir, 'split_train')
    validation_filename = _get_output_filename(dataset_dir, 'valid')

    #if tf.gfile.Exists(training_filename) and tf.gfile.Exists(validation_filename):
    #  print('Dataset files already exist. Exiting without re-creating them.')
    #  return

    # First, process the validation data:
    if tf.gfile.Exists(validation_filename):
        print(f'{validation_filename} already exist. Exiting without re-creating them.')
    else:
        with tf.python_io.TFRecordWriter(validation_filename) as tfrecord_writer:
            data_filename = os.path.join(dataset_dir, dataset_dir, _VALID_DATA_FILENAME)
            labels_filename = os.path.join(dataset_dir, _VALID_LABELS_FILENAME)
            # _add_to_valid_tfrecord(data_filename, labels_filename, 5000, tfrecord_writer)  # 500000
            print('>> Start valid-set tfrecord generation ...')
            _add_to_tfrecord(data_filename, 50000, tfrecord_writer)

    # Next, process the training data:
    if tf.gfile.Exists(training_filename):
        print(f'{validation_filename} already exist. Exiting without re-creating them.')
    else:
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
            data_filename = os.path.join(dataset_dir, dataset_dir, _TRAIN_DATA_FILENAME)
            labels_filename = os.path.join(dataset_dir, _LABELS_FILENAME)
            # _add_to_train_tfrecord(data_filename, labels_filename, 1281167, tfrecord_writer)  # 1281167
            print('>> Start train-set tfrecord generation ...')
            _add_to_tfrecord(data_filename, 1281167, tfrecord_writer, as_trainset=True)

    # Next, process the splitted train data:
    if tf.gfile.Exists(trainsplit_filename):
        print(f'{validation_filename} already exist. Exiting without re-creating them.')
    else:
        with tf.python_io.TFRecordWriter(trainsplit_filename) as tfrecord_writer:
            data_filename = os.path.join(dataset_dir, dataset_dir, _TRAIN_DATA_FILENAME)
            labels_filename = os.path.join(dataset_dir, _LABELS_FILENAME)
            # _add_to_trainsplit_tfrecord(data_filename, labels_filename, int(1281167*0.1), tfrecord_writer)
            print('>> Start train-split-set tfrecord generation ...')
            _add_to_tfrecord(data_filename, int(1281167 * 0.01), tfrecord_writer, as_trainset=False, as_splittrain=True)


    #_clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the ILSVRC12 dataset!')

if __name__ == '__main__':
    run()
