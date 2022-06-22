import tensorflow as tf
from dataset import ds_template
from dataset import tfrecord
import os

class Dataset(ds_template.DS_Template):
    def __init__(self, config):
        ds_template.DS_Template.__init__(self)
        self.config = config
        self.batch_size = self.config['Meta']['batchsize']

        self.raw_shape = [self.config['Dataset']['raw_img_size_x'], self.config['Dataset']['raw_img_size_y'],
                          self.config['Dataset']['img_depth']]
        self.img_shape = [self.config['Dataset']['img_size_x'], self.config['Dataset']['img_size_y'],
                          self.config['Dataset']['img_depth']]

        shuffles = self.config['Dataset']['shuffle']
        self.shuffle = {'train': shuffles[0], 'valid': shuffles[1], 'test': shuffles[2]}

        self.train_data = os.path.join(self.config['Dataset']['dataset_folder'], self.config['Dataset']['train_file'])
        self.valid_data = os.path.join(self.config['Dataset']['dataset_folder'], self.config['Dataset']['valid_file'])
        self.test_data = os.path.join(self.config['Dataset']['dataset_folder'], self.config['Dataset']['test_file'])

        self.pre_process_mode = self.config['Dataset']['pre_process_mode']

        self.ds_switch = tf.placeholder(dtype=tf.int32, name='dataset_switch')

        self.ds_name = self.config['Dataset']['name']
        # if self.ds_name == 'ImageNet2012':
        #     self._read_dataset_imagenet(self.shuffle)
        # # elif self.ds_name == 'CIFAR10':
        # #     self._read_dataset_cifar10()
        # else:
        #     self._read_dataset(self.shuffle)
        self._read_dataset(self.shuffle)

        self.train_augment = True
        self.test_augment = True

    def _read_dataset(self, shuffle):
        self.train_sample_image, self.train_sample_label = tfrecord.tfrecord_read_and_decode_single_example(
            [self.train_data], self.raw_shape, format=self.config['Dataset']['task_type'])
        img_train, self.lab_train = tfrecord.tfrecord_read_batch(self.train_sample_image, self.train_sample_label,
                                                                 self.batch_size, shuffle['train'],
                                                                 self.MAX_QUEUE_SIZE * self.batch_size,
                                                                 self.MIN_QUEUE_SIZE * self.batch_size)
        self.img_train = tf.reshape(img_train, shape=[-1, self.raw_shape[0], self.raw_shape[1], self.raw_shape[2]])

        self.valid_sample_image, self.valid_sample_label = tfrecord.tfrecord_read_and_decode_single_example(
            [self.valid_data], self.raw_shape, format=self.config['Dataset']['task_type'])
        img_valid, self.lab_valid = tfrecord.tfrecord_read_batch(self.valid_sample_image, self.valid_sample_label,
                                                                      self.batch_size, shuffle['valid'],
                                                                      self.MAX_QUEUE_SIZE * self.batch_size,
                                                                      self.MIN_QUEUE_SIZE * self.batch_size)
        self.img_valid = tf.reshape(img_valid, shape=[-1, self.raw_shape[0], self.raw_shape[1], self.raw_shape[2]])

        self.test_sample_image, self.test_sample_label = tfrecord.tfrecord_read_and_decode_single_example(
            [self.test_data], self.raw_shape, format=self.config['Dataset']['task_type'])
        img_test, self.lab_test = tfrecord.tfrecord_read_batch(self.test_sample_image, self.test_sample_label,
                                                                    self.batch_size, shuffle['test'],
                                                                    self.MAX_QUEUE_SIZE * self.batch_size,
                                                                    self.MIN_QUEUE_SIZE * self.batch_size)
        self.img_test = tf.reshape(img_test, shape=[-1, self.raw_shape[0], self.raw_shape[1], self.raw_shape[2]])

    def preprocess(self, image):
        image = tf.cast(image, dtype=tf.float32)

        if self.pre_process_mode == 'NORM':
            ret = tf.divide(tf.subtract(image, self.config['Dataset']['mean']), self.config['Dataset']['std'])
            # ret = tf.reshape(ret, [-1, self.raw_shape[1], self.raw_shape[0], self.raw_shape[2]])

        elif self.pre_process_mode == 'CHANNEL_NORM':
            mean = tf.constant([[[[self.config['Dataset']['mean_g'], self.config['Dataset']['mean_b'],
                                   self.config['Dataset']['mean_r']]]]], dtype=tf.float32)
            std = tf.constant([[[[self.config['Dataset']['std_g'], self.config['Dataset']['std_b'],
                                  self.config['Dataset']['std_r']]]]], dtype=tf.float32)
            image = tf.reshape(image, [-1, self.raw_shape[1], self.raw_shape[0], self.raw_shape[2]])
            ret = tf.divide(tf.subtract(image, mean), std)

        elif self.pre_process_mode == 'MEAN':
            ret = tf.subtract(image, self.config['Dataset']['mean'])
            # ret = tf.reshape(ret, [-1, self.raw_shape[1], self.raw_shape[0], self.raw_shape[2]])

        elif self.pre_process_mode == 'CHANNEL_MEAN':
            mean = tf.constant([[[[self.config['Dataset']['mean_g'], self.config['Dataset']['mean_b'],
                                   self.config['Dataset']['mean_r']]]]], dtype=tf.float32)
            image = tf.reshape(image, [-1, self.raw_shape[1], self.raw_shape[0], self.raw_shape[2]])
            ret = tf.subtract(image, mean)

        elif self.pre_process_mode == 'SIMPLE_NORM':
            ret = tf.divide(image, 255.0) - 0.5

        elif self.pre_process_mode == '0_1_NORM':
            ret = tf.divide(image, 255.0)

            # if self.ds_name == "ImageNet2012":
            #     mean = tf.constant([[[[0.485, 0.456, 0.406]]]], dtype=tf.float32)
            #     mean = tf.tile(mean, [self.batch_size, self.raw_shape[1], self.raw_shape[0], 1])
            #     std = tf.constant([[[[0.229, 0.224, 0.225]]]], dtype=tf.float32)
            #     std = tf.tile(std, [self.batch_size, self.raw_shape[1], self.raw_shape[0], 1])
            #     ret = tf.divide(tf.subtract(ret, mean), std)

        # elif self.pre_process_mode == 'PIXEL_MEAN':
        #     image = tf.reshape(image, [-1, self.raw_shape[1], self.raw_shape[0], self.raw_shape[2]])
        #     px_mean = np.array([np.load(self.mean_file)])
        #     px_mean = np.swapaxes(px_mean, 1, 3)
        #     px_mean = np.swapaxes(px_mean, 1, 2)
        #     mean = tf.constant(px_mean, dtype=tf.float32)
        #     ret = tf.subtract(image, mean)

        else:
            ret = tf.reshape(image, [-1, self.raw_shape[1], self.raw_shape[0], self.raw_shape[2]])

        return ret

    def _zero_pad(self, images):
        if self.config['Dataset']['pad_raw'] > 0:
            images = tf.image.resize_image_with_crop_or_pad(images,
                                                            self.config['Dataset']['img_size_y'] + self.config['Dataset']['pad_raw'],
                                                            self.config['Dataset']['img_size_x'] + self.config['Dataset']['pad_raw'])
        return images

    def _center_crop(self, images):
        if self.config['Dataset']['img_size_x'] < self.config['Dataset']['raw_img_size_y'] + self.config['Dataset']['pad_raw']:
            if self.config['Dataset']['img_size_y'] < self.config['Dataset']['raw_img_size_x'] + self.config['Dataset']['pad_raw']:
                fx = int(((self.config['Dataset']['raw_img_size_y'] + self.config['Dataset']['pad_raw']) -
                          self.config['Dataset']['img_size_y']) // 2)
                fy = int(((self.config['Dataset']['raw_img_size_x'] + self.config['Dataset']['pad_raw']) -
                          self.config['Dataset']['img_size_x']) // 2)

                images = tf.slice(images, begin=[0, fx, fy, 0],
                                  size=[-1, self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x'], -1])
        return images

    def _random_flip(self, images):
        images = [tf.image.random_flip_left_right(img) for img in images]
        return images

    def _random_crop(self, images):
        images = [tf.random_crop(tf.squeeze(img), [self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x'],
                                                   self.config['Dataset']['img_depth']]) for img in images]
        return images

    def _random_resized_crop(self, images):
        boxes = tf.random.uniform(shape=(5, 4))
        box_indices = tf.random.uniform(shape=(5,), minval=0,
                                        maxval=self.batch_size, dtype=tf.int32)
        images = tf.image.crop_and_resize(images, boxes, box_indices, (self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x']))

        return images

    def augment_img_valid(self, images):
        if self.config['Dataset']['task_type'] == 'classification':
            if self.ds_name != 'ImageNet2012':
                images = self._zero_pad(images)
                images = self._center_crop(images)
        elif self.config['Dataset']['task_type'] == 'segmentation':
            images = tf.image.resize_images(images, [self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x']],
                                            method=tf.image.ResizeMethod.BILINEAR)
        return images

    def augment_img_train(self, images):
        if self.config['Dataset']['task_type'] == 'classification' and self.config['Dataset']['name'] != 'MNIST':
            if self.ds_name != 'ImageNet2012':
                images = self._zero_pad(images)
                images = tf.split(images, self.batch_size)
                images = [tf.squeeze(img) for img in images]
                images = self._random_crop(images)
                images = self._random_flip(images)
                images = tf.stack(images)
        elif self.config['Dataset']['task_type'] == 'segmentation':
            images = tf.image.resize_images(images, [self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x']],
                                            method=tf.image.ResizeMethod.BILINEAR)
        return images

    def get_batch(self):
        self.train_mode = tf.constant(0, dtype=tf.int32)
        self.valid_mode = tf.constant(1, dtype=tf.int32)
        self.test_mode = tf.constant(2, dtype=tf.int32)

        if self.config['Dataset']['pre_process_mode'] == 'SIMPLE_NORM':
            images = tf.case([(tf.equal(self.ds_switch, self.valid_mode), lambda: self.preprocess(self.img_valid)),
                              (tf.equal(self.ds_switch, self.test_mode), lambda: self.preprocess(self.img_test))],
                             default=lambda: self.preprocess(self.img_train), exclusive=True)

        else:
            images = tf.case([(tf.equal(self.ds_switch, self.valid_mode), lambda: self.augment_img_valid(self.preprocess(self.img_valid))),
                              (tf.equal(self.ds_switch, self.test_mode), lambda: self.augment_img_valid(self.preprocess(self.img_test)))],
                             default=lambda: self.augment_img_train(self.preprocess(self.img_train)), exclusive=True)

        labels = tf.case([(tf.equal(self.ds_switch, self.valid_mode), lambda: self.lab_valid),
                         (tf.equal(self.ds_switch, self.test_mode), lambda: self.lab_test)],
                         default=lambda: self.lab_train, exclusive=True)

        return images, labels

    def get_switch(self):
        return self.ds_switch
