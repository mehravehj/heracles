'''
    Reference: https://github.com/DingZhenyuan/svhn-mat-tfrecords/blob/master/svhn_tyrecord.py
'''


import tensorflow as tf
import numpy as np
import sys
import os
import urllib
from scipy.io import loadmat


# Extract images and labels from dataset path
def data_set(name, ds_dir, num_sample_size=10000):
    if name == 'train':
        filename = ds_dir + "/train_32x32.mat"
    elif name == 'test':
        filename = ds_dir + "/test_32x32.mat"
    elif name == 'extra':
        filename = ds_dir + "/extra_32x32.mat"
    else:
        print("The name is wrong!")
    datadict = loadmat(filename)
    image = datadict['X']
    image = image.transpose((3, 0, 1, 2))

    label = datadict['y'].flatten()
    label[label == 10] = 0  # fix labels from 10--1 to 9--0

    return image, label


# data format
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# data format
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
    print(f'Successfully downloaded {filename} in {dataset_dir}.')


def convert_to_tfrecords(images, labels, fileName, image_format='png'.encode()):
    if os.path.exists(fileName):
        dirs = fileName.split('/')
        ds_folder = '/'.join(dirs[:-1])
        f_name = dirs[-1]
        print('{} already exists in {}.'.format(f_name, ds_folder))
    else:
        num_examples, rows, cols, depth = images.shape

        # write to tfrecord
        writer = tf.python_io.TFRecordWriter(fileName)
        for index in range(num_examples):
            sys.stdout.write('\r   Converting image %d/%d' % (index+1, num_examples))
            sys.stdout.flush()
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': _int64_feature(rows),
                'image/width': _int64_feature(cols),
                'image/format': _bytes_feature(image_format),
                'image/class/label': _int64_feature(int(labels[index])),
                'image/encoded': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()


# split dataset
def split_dataset(train_x, train_y, split_rate=0.1):
    num_images = len(train_x)

    cls_num_images = int(num_images * split_rate / 10)
    label_list = [[]] * 10

    for idx, lab in enumerate(train_y):
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

    train_x_sp = []
    train_y_sp = []
    valid_x_sp = []
    valid_y_sp = []
    for idx in f_list:
        x_tmp = train_x[idx]
        y_tmp = train_y[idx]
        valid_x_sp.append(x_tmp)
        valid_y_sp.append(y_tmp)

    for idx in rest_list:
        x_tmp = train_x[idx]
        y_tmp = train_y[idx]
        train_x_sp.append(x_tmp)
        train_y_sp.append(y_tmp)

    train_x_sp = np.array(train_x_sp)
    train_y_sp = np.array(train_y_sp)
    valid_x_sp = np.array(valid_x_sp)
    valid_y_sp = np.array(valid_y_sp)

    return train_x_sp, train_y_sp, valid_x_sp, valid_y_sp

def run():
    ds_dir = './data/SVHN'

    if not tf.gfile.Exists(ds_dir):
        tf.gfile.MakeDirs(ds_dir)

    if not tf.gfile.Exists(ds_dir + '/train_32x32.mat'):
        download_and_uncompress_tarball('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', ds_dir)
    if not tf.gfile.Exists(ds_dir + '/test_32x32.mat'):
        download_and_uncompress_tarball('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', ds_dir)
    # if not tf.gfile.Exists(ds_dir + '/extra_32x32.mat'):
    #     download_and_uncompress_tarball('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat', ds_dir)

    train_x, train_y = data_set('train', ds_dir, 73257)
    test_x, test_y = data_set('test', ds_dir, 26032)
    # extra_x, extra_y = data_set('extra', ds_dir, 10000)

    trainFileName = ds_dir + "/svhn_train.tfrecord"
    testFileName = ds_dir + "/svhn_test.tfrecord"
    # extraFileName = ds_dir + "/svhn_extra.tfrecord"

    # Convert to Examples and write the result to TFRecords.
    print('\n>> Generating Train Set:')
    convert_to_tfrecords(train_x, train_y, trainFileName)
    print('\n>> Generating Test Set:')
    convert_to_tfrecords(test_x, test_y, testFileName)
    # print('\n>> Generating Extra Set:')
    # convert_to_tfrecords(extra_x, extra_y, extraFileName)

    # Split train set for validation
    train_x_sp, train_y_sp, valid_x_sp, valid_y_sp = split_dataset(train_x, train_y, 0.1)

    train_split_fname = ds_dir + "/svhn_train_rest.tfrecord"
    valid_split_fname = ds_dir + "/svhn_train_holdout.tfrecord"

    # Convert to Examples and write the result to TFRecords.
    print('\n>> Generating Train-Rest Set:')
    convert_to_tfrecords(train_x_sp, train_y_sp, train_split_fname)
    print('\n>> Generating Train-Split Set:')
    convert_to_tfrecords(valid_x_sp, valid_y_sp, valid_split_fname)

    print('\nTF-Records generation finished.')


if __name__ == '__main__':
    run()

