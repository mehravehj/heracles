import tensorflow as tf
from nn import layer, modules
from ops import loss_op
from ops import train_op
from ops import eval_op
from ops.learning_rate import learning_rate
from nn.hw_structure import HW_Structure
import numpy as np


class vgg16_bn(object):
    def __init__(self, params, ds_switch):
        self.params = params
        self.ds_switch = ds_switch

        self.is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

        self.lr = learning_rate(params=self.params)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.drop_out = self.params['Meta']['dropout']
        self.bsize = self.params['Meta']['batchsize']

    def training(self, lab):
        self.total_loss, self.pred_loss, self.reg_loss = loss_op.loss_op(logits=self.logits, labels=lab, params=self.params)
        self.train_op = train_op.get_train_op(model=self, loss=self.total_loss)

    def evaluation(self, lab, attack_on=False):
        self.acc_1, self.acc_5, self.preds, self.y = eval_op.get_eval_op(logits=self.logits, labels=lab, batch_size=self.bsize)
        self.eval_op = [self.acc_1, self.acc_5, self.preds, lab, self.total_loss, self.reg_loss]
        self.tb_merged = tf.summary.merge_all()

    def hw_struct(self):
        self.layer_weights = [w for w in tf.get_collection(tf.GraphKeys.WEIGHTS) if
                         (w.name.find('conv') != -1 or w.name.find('fc') != -1) and w.name.find('bias') == -1]
        self.bias = [w for w in tf.get_collection(tf.GraphKeys.BIASES)]
        self.f_maps = [fp for fp in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if fp.name.find('f_map') != -1 and fp.name.find('/inference/') != -1]
        self.acts = [act for act in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if act.name.find('f_map') == -1]
        self.hw_structs = HW_Structure(self.layer_weights, self.f_maps, self.acts)

    def inference(self, img, reuse=False):
        num_class = self.params['Dataset']['num_classes']
        filters =   [64,    64,     128,    128,    256,    256,    256,    512,    512,    512,    512,    512,    512,    256,    256,    num_class]
        kernels =   [3,     3,      3,      3,      3,      3,      3,      3,      3,      3,      3,      3,      3,      1,      1,      1]
        dp_rates =  [0.3,   None,   0.4,    None,   0.4,    0.4,    None,   0.4,    0.4,    None,   0.4,    0.4,    None,   0.5,    0.5,    0.5]
        max_pools = [None,  2,      None,   2,      None,   None,   2,      None,   None,   2,      None,   None,   2,      None,   None,   None]
        strides =   [1,     1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      None,   None,   None]

        prune_reg = self.params['Prune_config']['prune_reg']

        with tf.variable_scope('inference', reuse=reuse):

            with tf.name_scope('input'):
                if self.params['Dataset']['pre_process_mode'] == '':
                     self.inputs = tf.map_fn(lambda image: tf.image.per_image_standardization(image), img)
                else:
                    self.inputs = img

            with tf.variable_scope('conv1_1'):
                layer_idx = 0
                conv1_1_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv1_1 = layer.get_conv_layer(img, co=filters[layer_idx], k=kernels[layer_idx], bias=False,
                                                     stride=strides[layer_idx], params=conv1_1_params, prune_reg=prune_reg)

                self._conv1_1 = layer.get_bn_layer(self._conv1_1)
                self._conv1_1 = layer.get_relu(self._conv1_1)

                if self.drop_out:
                    self._conv1_1 = layer.get_dropout(self._conv1_1, dp_rates[layer_idx])

            with tf.variable_scope('conv1_2'):
                layer_idx += 1
                conv1_2_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv1_2 = layer.get_conv_layer(self._conv1_1, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv1_2_params, prune_reg=prune_reg)
                self._conv1_2 = layer.get_bn_layer(self._conv1_2)
                self._conv1_2 = layer.get_relu(self._conv1_2)

                # 'max_pool_1':
                self._max_pool_1 = layer.get_max_pool(self._conv1_2, k=max_pools[layer_idx], stride=2, pad='VALID')

            with tf.variable_scope('conv2_1'):
                layer_idx += 1
                conv2_1_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv2_1 = layer.get_conv_layer(self._max_pool_1, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv2_1_params, prune_reg=prune_reg)
                self._conv2_1 = layer.get_bn_layer(self._conv2_1)
                self._conv2_1 = layer.get_relu(self._conv2_1)

                if self.drop_out:
                    self._conv2_1 = layer.get_dropout(self._conv2_1, dp_rates[layer_idx])

            with tf.variable_scope('conv2_2'):
                layer_idx += 1
                conv2_2_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv2_2 = layer.get_conv_layer(self._conv2_1, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv2_2_params, prune_reg=prune_reg)
                self._conv2_2 = layer.get_bn_layer(self._conv2_2)
                self._conv2_2 = layer.get_relu(self._conv2_2)

                # 'max_pool_2':
                self._max_pool_2 = layer.get_max_pool(self._conv2_2, k=max_pools[layer_idx], stride=2, pad='VALID')

            with tf.variable_scope('conv3_1'):
                layer_idx += 1
                conv3_1_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv3_1 = layer.get_conv_layer(self._max_pool_2, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv3_1_params, prune_reg=prune_reg)
                self._conv3_1 = layer.get_bn_layer(self._conv3_1)
                self._conv3_1 = layer.get_relu(self._conv3_1)

                if self.drop_out:
                    self._conv3_1 = layer.get_dropout(self._conv3_1, dp_rates[layer_idx])

            with tf.variable_scope('conv3_2'):
                layer_idx += 1
                conv3_2_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv3_2 = layer.get_conv_layer(self._conv3_1, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv3_2_params, prune_reg=prune_reg)
                self._conv3_2 = layer.get_bn_layer(self._conv3_2)
                self._conv3_2 = layer.get_relu(self._conv3_2)

                if self.drop_out:
                    self._conv3_2 = layer.get_dropout(self._conv3_2, dp_rates[layer_idx])

            with tf.variable_scope('conv3_3'):
                layer_idx += 1
                conv3_3_params = {'weight': {'init': {'type': 'he'}},
                                  'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv3_3 = layer.get_conv_layer(self._conv3_2, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv3_3_params, prune_reg=prune_reg)
                self._conv3_3 = layer.get_bn_layer(self._conv3_3)
                self._conv3_3 = layer.get_relu(self._conv3_3)

                # 'max_pool_3':
                self._max_pool_3 = layer.get_max_pool(self._conv3_3, k=max_pools[layer_idx], stride=2, pad='VALID')

            with tf.variable_scope('conv4_1'):
                layer_idx += 1
                conv4_1_params = {
                    'weight': {'init': {'type': 'he'}},
                    'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv4_1 = layer.get_conv_layer(self._max_pool_3, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv4_1_params, prune_reg=prune_reg)
                self._conv4_1 = layer.get_bn_layer(self._conv4_1)
                self._conv4_1 = layer.get_relu(self._conv4_1)

                if self.drop_out:
                    self._conv4_1 = layer.get_dropout(self._conv4_1, dp_rates[layer_idx])

            with tf.variable_scope('conv4_2'):
                layer_idx += 1
                conv4_2_params = {
                    'weight': {'init': {'type': 'he'}},
                    'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv4_2 = layer.get_conv_layer(self._conv4_1, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv4_2_params, prune_reg=prune_reg)
                self._conv4_2 = layer.get_bn_layer(self._conv4_2)
                self._conv4_2 = layer.get_relu(self._conv4_2)

                if self.drop_out:
                    self._conv4_2 = layer.get_dropout(self._conv4_2, dp_rates[layer_idx])

            with tf.variable_scope('conv4_3'):
                layer_idx += 1
                conv4_3_params = {
                    'weight': {'init': {'type': 'he'}},
                    'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv4_3 = layer.get_conv_layer(self._conv4_2, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv4_3_params, prune_reg=prune_reg)
                self._conv4_3 = layer.get_bn_layer(self._conv4_3)
                self._conv4_3 = layer.get_relu(self._conv4_3)

                # 'max_pool_4':
                self._max_pool_4 = layer.get_max_pool(self._conv4_3, k=max_pools[layer_idx], stride=2, pad='VALID')

            with tf.variable_scope('conv5_1'):
                layer_idx += 1
                conv5_1_params = {
                    'weight': {'init': {'type': 'he'}},
                    'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv5_1 = layer.get_conv_layer(self._max_pool_4, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv5_1_params, prune_reg=prune_reg)
                self._conv5_1 = layer.get_bn_layer(self._conv5_1)
                self._conv5_1 = layer.get_relu(self._conv5_1)

                if self.drop_out:
                    self._conv5_1 = layer.get_dropout(self._conv5_1, dp_rates[layer_idx])

            with tf.variable_scope('conv5_2'):
                layer_idx += 1
                conv5_2_params = {
                    'weight': {'init': {'type': 'he'}},
                    'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv5_2 = layer.get_conv_layer(self._conv5_1, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv5_2_params, prune_reg=prune_reg)
                self._conv5_2 = layer.get_bn_layer(self._conv5_2)
                self._conv5_2 = layer.get_relu(self._conv5_2)

                if self.drop_out:
                    self._conv5_2 = layer.get_dropout(self._conv5_2, dp_rates[layer_idx])

            with tf.variable_scope('conv5_3'):
                layer_idx += 1
                conv5_3_params = {
                    'weight': {'init': {'type': 'he'}},
                    'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv5_3 = layer.get_conv_layer(self._conv5_2, co=filters[layer_idx], k=kernels[layer_idx],
                                                     bias=False, stride=strides[layer_idx], params=conv5_3_params, prune_reg=prune_reg)
                self._conv5_3 = layer.get_bn_layer(self._conv5_3)
                self._conv5_3 = layer.get_relu(self._conv5_3)

                # 'avg_pool_5'
                self._avg_pool_5 = layer.get_avg_pool(self._conv5_3, k=1, stride=1, pad='VALID')

            with tf.variable_scope('fc1'):
                layer_idx += 1
                fc_params = {'weight': {'init': {'type': 'null'}},
                              'bias': {'init': {'type': 'const', 'val': 0.0}}}

                if self.drop_out:
                    self._fc1_input = layer.get_dropout(self._avg_pool_5, dp_rates[layer_idx])
                else:
                    self._fc1_input = self._avg_pool_5

                # Use torch flatten mode for Hydra model
                self._fc1_input = tf.transpose(self._fc1_input, (0,3,1,2))

                self._fc1_input = tf.reshape(self._fc1_input, shape=[self._fc1_input.shape.as_list()[0], -1])
                self._fc1 = layer.get_fc_layer(self._fc1_input, co=filters[layer_idx], params=fc_params, prune_reg=prune_reg)
                self._fc1 = layer.get_relu(self._fc1)

            with tf.variable_scope('fc2'):
                layer_idx += 1
                fc_params = {'weight': {'init': {'type': 'null'}},
                              'bias': {'init': {'type': 'const', 'val': 0.0}}}

                if self.drop_out:
                    self._fc2_input = layer.get_dropout(self._fc1, dp_rates[layer_idx])
                else:
                    self._fc2_input = self._fc1

                self._fc2 = layer.get_fc_layer(self._fc2_input, co=filters[layer_idx], params=fc_params, prune_reg=prune_reg)
                self._fc2 = layer.get_relu(self._fc2)

            with tf.variable_scope('fc3'):
                layer_idx += 1
                fc_params = {'weight': {'init': {'type': 'null'}},
                              'bias': {'init': {'type': 'const', 'val': 0.0}}}

                if self.drop_out:
                    self._fc3_input = layer.get_dropout(self._fc2, dp_rates[layer_idx])
                else:
                    self._fc3_input = self._fc2

                self._fc3 = layer.get_fc_layer(self._fc3_input, co=filters[layer_idx], params=fc_params, prune_reg=prune_reg)

        self.logits = self._fc3

        return self.logits
