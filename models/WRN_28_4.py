import tensorflow as tf
from nn import layer, modules
from ops import loss_op
from ops import train_op
from ops import eval_op
from ops.learning_rate import learning_rate
from nn.hw_structure import HW_Structure
import numpy as np


class wide_resnet_28_4(object):
    def __init__(self, params, ds_switch):
        self.params = params
        self.ds_switch = ds_switch

        self.is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

        self.lr = learning_rate(params=self.params)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def training(self, lab):
        self.total_loss, self.pred_loss, self.reg_loss = loss_op.loss_op(logits=self.logits, labels=lab, params=self.params)
        self.train_op = train_op.get_train_op(model=self, loss=self.total_loss)

    def evaluation(self, lab):
        bsize = self.params['Meta']['batchsize']
        self.acc_1, self.acc_5, self.preds, self.y = eval_op.get_eval_op(logits=self.logits, labels=lab, batch_size=bsize)
        self.eval_op = [self.acc_1, self.acc_5, self.preds, lab, self.total_loss, self.reg_loss]
        self.tb_merged = tf.summary.merge_all()

    def hw_struct(self):
        self.layer_weights = [w for w in tf.get_collection(tf.GraphKeys.WEIGHTS) if
                              (w.name.find('conv') != -1 or w.name.find('fc') != -1) and w.name.find('bias') == -1]
        self.f_maps = [fp for fp in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if fp.name.find('f_map') != -1]
        self.acts = [act for act in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if act.name.find('f_map') == -1]
        self.hw_structs = HW_Structure(self.layer_weights, self.f_maps, self.acts)

    def inference(self, img, reuse=False):
        # TODO: 28=3*big_block + 2*shortcut + input_layer + last_fc_layer (bigblock=4*baseblock)
        num_class = self.params['Dataset']['num_classes']

        wide_factor = 4

        filters = [16, 16*wide_factor, 32*wide_factor, 64*wide_factor, num_class]
        kernels = [3, 3, 3, 3]
        strides = [1, 1, 2, 2]

        prune_reg = self.params['Prune_config']['prune_reg']

        with tf.variable_scope('inference', reuse=reuse):

            with tf.name_scope('input'):
                if self.params['Dataset']['pre_process_mode'] == '':
                     self.inputs = tf.map_fn(lambda image: tf.image.per_image_standardization(image), img)
                else:
                    self.inputs = img

            ''' =========== conv1 =========== '''

            with tf.variable_scope('conv1'):
                w_init_std = np.sqrt(2.0 / kernels[0] / kernels[0] / filters[0])
                conv1_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv1 = layer.get_conv_layer(self.inputs, co=filters[0], k=kernels[0], stride=strides[0],
                                                   params=conv1_train_params, prune_reg=prune_reg)
                if self._conv1 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._conv1)

            ''' =========== conv2_x =========== '''

            with tf.variable_scope('conv2_1'):
                w_init_std = np.sqrt(2.0 / kernels[1] / kernels[1] / filters[1])
                conv2_1_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv2_1 = modules._residual_block_first_preact(self._conv1, co=filters[1], k=kernels[1], stride=strides[1],
                                                                   params=conv2_1_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv2_2'):
                w_init_std = np.sqrt(2.0 / kernels[1] / kernels[1] / filters[1])
                conv2_2_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv2_2 = modules._residual_block_preact(self._conv2_1, co=filters[1], k=kernels[1], stride=1,
                                                        params=conv2_2_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv2_3'):
                w_init_std = np.sqrt(2.0 / kernels[1] / kernels[1] / filters[1])
                conv2_3_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv2_3 = modules._residual_block_preact(self._conv2_2, co=filters[1], k=kernels[1], stride=1,
                                                        params=conv2_3_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv2_4'):
                w_init_std = np.sqrt(2.0 / kernels[1] / kernels[1] / filters[1])
                conv2_4_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv2_4 = modules._residual_block_preact(self._conv2_3, co=filters[1], k=kernels[1], stride=1,
                                                        params=conv2_4_train_params, prune_reg=prune_reg)


            ''' =========== conv3_x =========== '''

            with tf.variable_scope('conv3_1'):
                w_init_std = np.sqrt(2.0 / kernels[2] / kernels[2] / filters[2])
                conv3_1_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv3_1 = modules._residual_block_first_preact(self._conv2_4, co=filters[2], k=kernels[2], stride=strides[2],
                                                              params=conv3_1_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv3_2'):
                w_init_std = np.sqrt(2.0 / kernels[2] / kernels[2] / filters[2])
                conv3_2_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv3_2 = modules._residual_block_preact(self._conv3_1, co=filters[2], k=kernels[2], stride=1,
                                                        params=conv3_2_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv3_3'):
                w_init_std = np.sqrt(2.0 / kernels[2] / kernels[2] / filters[2])
                conv3_3_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv3_3 = modules._residual_block_preact(self._conv3_2, co=filters[2], k=kernels[2], stride=1,
                                                        params=conv3_3_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv3_4'):
                w_init_std = np.sqrt(2.0 / kernels[2] / kernels[2] / filters[2])
                conv3_4_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv3_4 = modules._residual_block_preact(self._conv3_3, co=filters[2], k=kernels[2], stride=1,
                                                        params=conv3_4_train_params, prune_reg=prune_reg)

            ''' =========== conv4_x =========== '''

            with tf.variable_scope('conv4_1'):
                w_init_std = np.sqrt(2.0 / kernels[3] / kernels[3] / filters[3])
                conv4_1_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv4_1 = modules._residual_block_first_preact(self._conv3_4, co=filters[3], k=kernels[3], stride=strides[3],
                                                                    params=conv4_1_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv4_2'):
                w_init_std = np.sqrt(2.0 / kernels[3] / kernels[3] / filters[3])
                conv4_2_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv4_2 = modules._residual_block_preact(self._conv4_1, co=filters[3], k=kernels[3], stride=1,
                                                        params=conv4_2_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv4_3'):
                w_init_std = np.sqrt(2.0 / kernels[3] / kernels[3] / filters[3])
                conv4_3_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv4_3 = modules._residual_block_preact(self._conv4_2, co=filters[3], k=kernels[3], stride=1,
                                                        params=conv4_3_train_params, prune_reg=prune_reg)

            with tf.variable_scope('conv4_4'):
                w_init_std = np.sqrt(2.0 / kernels[3] / kernels[3] / filters[3])
                conv4_4_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._conv4_4 = modules._residual_block_preact(self._conv4_3, co=filters[3], k=kernels[3], stride=1,
                                                        params=conv4_4_train_params, prune_reg=prune_reg)


            ''' =========== fc_logits =========== '''

            with tf.variable_scope('fc'):
                num_class = self.params['Dataset']['num_classes']
                w_init_std = np.sqrt(1.0/float(num_class))
                fc_train_params = {
                    'weight': {'init': {'type': 'he', 'std': w_init_std}, 'lr_mult': 1, 'decay_mult': 1},
                    'bias': {'init': {'type': 'const', 'val': 0.0}, 'lr_mult': 2, 'decay_mult': 0}}

                self._fc_pre_bn = layer.get_bn_layer(self._conv4_4, name='bn')
                if self._fc_pre_bn not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._fc_pre_bn)

                self._fc_pre_relu = layer.get_relu(self._fc_pre_bn, name='Relu')
                if self._fc_pre_relu not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._fc_pre_relu)

                self._fc_pre = layer.get_avg_pool(self._fc_pre_relu, k=8, stride=8)  # As in torch.nn.functional.avg_pool2d, default stride = kernel_size
                self._fc = tf.reshape(self._fc_pre, shape=[self._fc_pre.shape.as_list()[0], -1])

                self._fc = layer.get_fc_layer(self._fc, co=self.params['Dataset']['num_classes'],
                                              params=fc_train_params, prune_reg=prune_reg)

        self.logits = self._fc

        return self.logits
