import tensorflow as tf
from nn import layer, modules
from ops import loss_op
from ops import train_op
from ops import eval_op
from ops.learning_rate import learning_rate
from nn.hw_structure import HW_Structure
import numpy as np

class resnet50(object):
    def __init__(self, params, ds_switch):
        self.params = params
        self.ds_switch = ds_switch
        self.ds_name = self.params['Dataset']['name']

        self.batch_size = self.params['Meta']['batchsize']

        self.raw_shape = [self.params['Dataset']['raw_img_size_x'], self.params['Dataset']['raw_img_size_y'],
                          self.params['Dataset']['img_depth']]
        self.img_shape = [self.params['Dataset']['img_size_x'], self.params['Dataset']['img_size_y'],
                          self.params['Dataset']['img_depth']]

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

    def inference(self, img, wide=16):
        num_class = self.params['Dataset']['num_classes']

        blocks = [3, 4, 6, 3]
        layers = 3
        filters = [4*wide,
                   [4*wide, 4*wide, 16*wide],
                   [8*wide, 8*wide, 32*wide],
                   [16*wide, 16*wide, 64*wide],
                   [32*wide, 32*wide, 128*wide],
                   num_class]
        kernels = [7, [1, 3, 1], [1, 3, 1], [1, 3, 1], [1, 3, 1]]
        strides = [2, [1, 1, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1]]


        prune_reg = self.params['Prune_config']['prune_reg']

        with tf.variable_scope('inference', reuse=False):

            with tf.name_scope('input'):
                if self.params['Dataset']['pre_process_mode'] == '':
                     self.inputs = tf.map_fn(lambda image: tf.image.per_image_standardization(image), img)
                else:
                    self.inputs = img

                if self.ds_name == "ImageNet2012":
                    mean = tf.constant([[[[0.485, 0.456, 0.406]]]], dtype=tf.float32)
                    mean = tf.tile(mean, [self.batch_size, self.img_shape[1], self.img_shape[0], 1])
                    std = tf.constant([[[[0.229, 0.224, 0.225]]]], dtype=tf.float32)
                    std = tf.tile(std, [self.batch_size, self.img_shape[1], self.img_shape[0], 1])
                    self.inputs = tf.divide(tf.subtract(self.inputs, mean), std)

            ''' =========== conv1 =========== '''

            with tf.variable_scope('conv1'):
                w_init_std = np.sqrt(2.0 / kernels[0] / kernels[0] / filters[0])
                conv1_train_params = {'weight': {'init': {'type': 'he'}},
                                      'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._conv1 = layer.get_conv_layer(self.inputs, co=filters[0], k=kernels[0], stride=strides[0],
                                                   pad=3, params=conv1_train_params, prune_reg=prune_reg)
                if self._conv1 not in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._conv1)

                self._conv1_bn =layer.get_bn_layer(self._conv1)
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._conv1_bn)

                self._conv1_relu = layer.get_relu(self._conv1_bn)
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._conv1_relu)

            self._maxpool_1 = layer.get_max_pool(self._conv1_relu, k=3, stride=2, pad=1)

            train_params = {'weight': {'init': {'type': 'he'}},
                            'bias': {'init': {'type': 'const', 'val': 0.0}}}

            ''' =========== Conv Res-Blocks =========== '''
            # TODO: Big trouble from conv4_3
            m_in = self._maxpool_1
            for b_id, block in enumerate(blocks):
                for m in range(block):
                    scope_name = 'conv{}_{}'.format(b_id+2, m+1)
                    with tf.variable_scope(scope_name):
                        m_out = '_' + scope_name
                        if m == 0:
                            m_val = modules._residual_block_bottleneck(m_in, co=filters[b_id+1], k=kernels[b_id+1],
                                                            stride=strides[b_id+1], params=train_params,
                                                            layers=layers, prune_reg=prune_reg)
                        else:
                            m_val = modules._residual_block_bottleneck(m_in, co=filters[b_id+1], k=kernels[b_id+1],
                                                            stride=[1, 1, 1], params=train_params,
                                                            layers=layers, prune_reg=prune_reg)
                        setattr(self, m_out, m_val)

                    m_in = getattr(self, m_out)

            self._conv_out = getattr(self, m_out)

            ''' =========== fc_logits =========== '''

            # TODO: Before FC layer, there should be an avg_pool layer with stride=4

            with tf.variable_scope('fc'):
                fc_train_params = {'weight': {'init': {'type': 'null'}},
                                   'bias': {'init': {'type': 'const', 'val': 0.0}}}

                self._fc_pre = layer.get_avg_pool(self._conv_out, k=7, stride=7)
                self._fc = tf.reshape(self._fc_pre, shape=[self._fc_pre.shape.as_list()[0], -1])
                self._fc = layer.get_fc_layer(self._fc, co=self.params['Dataset']['num_classes'],
                                              params=fc_train_params, prune_reg=prune_reg)

        self.logits = self._fc

        return self.logits
