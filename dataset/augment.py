import tensorflow as tf

class Augment():
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

    def augment_img_valid(self, images):
        if self.config['Dataset']['task_type'] == 'classification':
            images = self._zero_pad(images, 'valid')
            images = self._center_crop(images)
        elif self.config['Dataset']['task_type'] == 'segmentation':
            images = tf.image.resize_images(images, [self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x']], method=tf.image.ResizeMethod.BILINEAR)
        return images

    def augment_img_train(self, images):
        if self.config['Dataset']['task_type'] == 'classification' and self.config['Dataset']['name'] != 'MNIST':
            images = self._zero_pad(images, 'train')
            images = tf.split(images, self.batch_size)
            images = [tf.squeeze(img) for img in images]
            images = self._random_crop(images)
            images = self._random_flip(images)
            images = tf.stack(images)
        elif self.config['Dataset']['task_type'] == 'segmentation':
            images = tf.image.resize_images(images, [self.config['Dataset']['img_size_y'], self.config['Dataset']['img_size_x']], method=tf.image.ResizeMethod.BILINEAR)
        return images
