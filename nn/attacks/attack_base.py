import tensorflow as tf
import numpy as np
from nn.attacks import _attack_template
import math


class Attack_Base(_attack_template.Attack_Template):
    def __init__(self, img, lab, params, ds_switch):
        """ Adversarial Attack Methods implementation in Tensorflow.
                        :param img_var: 2D or 4D input tensor.
                        :param lab_var: 1D tensor with all expected labels
                        :param noise_var: 2D or 4D tensor (same shape as img_var)
                        :param epsilon: Upper & Lower strength limitation of attack noise.
                        :param clip_min: Min clip value for output.
                        :param clip_max: Max clip value for output.
        """

        _attack_template.Attack_Template.__init__(self)

        self.params = params

        self.img_input = img
        self.lab_input = lab

        self.ds_switch = ds_switch

        self.aa_on = tf.placeholder(tf.bool, name='aa_input_switch')
        self.aa_init_on = tf.placeholder(tf.bool, name='aa_init_switch')
        self.acc_switch = tf.placeholder(tf.bool, name='acc_switch')
        self.iter = tf.placeholder(tf.float32, name='attack_iter')
        self.run_eval = tf.placeholder_with_default(0, (), name='eval_switch')

        self.aa_method = self.convert_name(params['Attack_config']['method'])
        self.rand_init = self.params['Attack_config']['rand_init']

        self.batchsize = self.img_input.get_shape().as_list()[0]

        self.clip_min, self.clip_max = self.get_clip_lims()
        self.init_params()
        self.epsilon = self.params['Attack_config']['epsilon']
        self.stepsize = self.params['Attack_config']['stepsize']

        self.eval_eps = self.params['Attack_config']['eval_eps']
        self.eval_stepsize = self.params['Attack_config']['eval_stepsize']

        self.init_attack_vars()

        self.img = self.get_preprocess()

        self.kl_robust = self.get_kl_robustness()
        self.xent_robust = self.get_xent_robustness()

    def init_params(self):
        if self.params['Dataset']['pre_process_mode'] == '0_1_NORM':
            if self.params['Attack_config']['epsilon'] >= 1.0:
                self.params['Attack_config']['epsilon'] /= 255.0
            if self.params['Attack_config']['stepsize'] >= 1.0:
                self.params['Attack_config']['stepsize'] /= 255.0

            if self.params['Attack_config']['eval_eps'] >= 1.0:
                self.params['Attack_config']['eval_eps'] /= 255.0
            if self.params['Attack_config']['eval_stepsize'] >= 1.0:
                self.params['Attack_config']['eval_stepsize'] /= 255.0

        elif self.params['Dataset']['pre_process_mode'] == 'CHANNEL_NORM':
            clip_dist = self.clip_max - self.clip_min
            if self.params['Attack_config']['epsilon'] >= 1.0:
                self.params['Attack_config']['epsilon'] /= 255.0/clip_dist
            if self.params['Attack_config']['stepsize'] >= 1.0:
                self.params['Attack_config']['stepsize'] /= 255.0/clip_dist

            if self.params['Attack_config']['eval_eps'] >= 1.0:
                self.params['Attack_config']['eval_eps'] /= 255.0/clip_dist
            if self.params['Attack_config']['eval_stepsize'] >= 1.0:
                self.params['Attack_config']['eval_stepsize'] /= 255.0/clip_dist

        else:
            if self.params['Attack_config']['epsilon'] < 1.0:
                self.params['Attack_config']['epsilon'] *= 255.0
            if self.params['Attack_config']['stepsize'] < 1.0:
                self.params['Attack_config']['stepsize'] *= 255.0

            if self.params['Attack_config']['eval_eps'] < 1.0:
                self.params['Attack_config']['eval_eps'] *= 255.0
            if self.params['Attack_config']['eval_stepsize'] < 1.0:
                self.params['Attack_config']['eval_stepsize'] *= 255.0

    def init_attack_vars(self):
        # Initialize 2 global variables which are used to save the images & labels
        self.img_shape = self.img_input.get_shape().as_list()
        self.lab_shape = self.lab_input.get_shape().as_list()

        self.img_org = tf.get_variable(name='origin_img',
                                           shape=self.img_shape,
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer(),
                                           trainable=False,
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.img_var = tf.get_variable(name='input_img',
                                       shape=self.img_shape,
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.lab_org = tf.get_variable(name='origin_lab',
                                       shape=self.lab_shape,
                                       dtype=tf.int64,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.lab_var = tf.get_variable(name='input_lab',
                                       shape=self.lab_shape,
                                       dtype=tf.int64,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        trainable = True if self.aa_method in [self.CW, self.SPSA] else False

        self.noise_var = tf.get_variable(name='adv_noise',
                                         shape=self.img_shape,
                                         dtype=tf.float32,
                                         trainable=trainable,
                                         initializer=tf.zeros_initializer(),
                                         collections=[tf.GraphKeys.LOCAL_VARIABLES])

        # self.acc_dims = self.params['Attack_config']['iters'] + 1

    # Generate target labels from lab_input
    def get_target_labs(self):
        mask = tf.zeros(shape=tf.shape(self.lab_input), dtype=tf.int64)

        if self.params['Attack_config']['target_labs'] == []:
            lab_target = tf.random_uniform(self.lab_shape, minval=0, maxval=self.params['Dataset']['num_classes']-1, dtype=tf.int64)
        else:
            for fool_lab in self.params['Attack_config']['target_labs']:
                lab_mask_temp = tf.cast(tf.equal(self.lab_input, fool_lab), dtype=tf.int64)
                mask = tf.bitwise.bitwise_or(mask, lab_mask_temp)

            org_change = tf.ones_like(self.lab_input) * fool_lab
            neg_mask = tf.negative(mask) + 1
            lab_target = tf.multiply(org_change, neg_mask) + mask * self.params['Attack_config']['target']

        lab_assign = self.lab_var.assign(lab_target)

        return lab_assign

    def get_init_noise(self):
        if self.rand_init:
            rand_init = tf.random_uniform(shape=self.noise_var.shape.as_list(), minval=-self.epsilon, maxval=self.epsilon)
            eval_init = tf.random_uniform(shape=self.noise_var.shape.as_list(), minval=-self.eval_eps, maxval=self.eval_eps)
        else:
            rand_init = tf.zeros_like(self.noise_var)
            eval_init = tf.zeros_like(self.noise_var)

        init_noise = tf.case([(tf.equal(self.run_eval, 1), lambda: eval_init)], default=lambda: rand_init)

        noise_init_op = self.noise_var.assign(init_noise)

        return noise_init_op

    def get_adv_input(self):

        self.target_on = self.params['Attack_config']['target_on']
        self.if_target = tf.logical_and(self.target_on, tf.logical_not(self.aa_on))

        # Ops: Update model inputs with attacked images and their labels

        assign_img_org = tf.case([(tf.equal(self.aa_on, True), lambda: self.img_org)],
                                 default=lambda: self.img_org.assign(self.img_input))

        assign_img_var = tf.case([(tf.equal(self.aa_on, True), lambda: self.img_var)],
                                 default=lambda: self.img_var.assign(self.img_input))

        assign_lab_var = tf.case([(self.if_target, lambda: self.get_target_labs()),
                                  (tf.equal(self.aa_on, True), lambda: self.lab_var)],
                                 default=lambda: self.lab_var.assign(self.lab_input))

        assign_lab_org = tf.case([(tf.equal(self.aa_on, True), lambda: self.lab_org)],
                                 default=lambda: self.lab_org.assign(self.lab_input))

        assign_init_noise = tf.case([(tf.equal(self.aa_on, True), lambda: self.noise_var)],
                                    default=lambda: self.get_init_noise())

        return assign_img_var, assign_img_org, assign_lab_var, assign_lab_org, assign_init_noise

    # Get epsilon
    def get_stepsize(self):
        return self.stepsize

    # Get attack switch
    def get_attack_switch(self):
        return self.aa_on

    # Get attack init switch
    def get_attack_init_switch(self):
        return self.aa_init_on

    # Get attack iter switch
    def get_attack_iter(self):
        return self.iter

    # Get min. & max. value of image
    def get_clip_lims(self):
        prepro_mode = self.params['Dataset']['pre_process_mode']

        if prepro_mode == 'NORM':
            clip_min = (0.0 - self.params['Dataset']['mean']) / self.params['Dataset']['std']
            clip_max = (255.0 - self.params['Dataset']['mean']) / self.params['Dataset']['std']
        elif prepro_mode == 'CHANNEL_NORM':
            mean = np.array([self.params['Dataset']['mean_r'], self.params['Dataset']['mean_g'], self.params['Dataset']['mean_b']])
            std = np.array([self.params['Dataset']['std_r'], self.params['Dataset']['std_g'], self.params['Dataset']['std_b']])
            clip_min = max((0.0 - mean)/std)
            clip_max = min((255.0 - mean)/std)
        elif prepro_mode == 'MEAN':
            clip_min = 0.0 - self.params['Dataset']['mean']
            clip_max = 255.0 - self.params['Dataset']['mean']
        elif prepro_mode == 'CHANNEL_MEAN':
            mean = tf.constant([self.params['Dataset']['mean_r'], self.params['Dataset']['mean_g'], self.params['Dataset']['mean_b']], dtype=tf.float32)
            clip_min = tf.reduce_max(0.0 - mean)
            clip_max = tf.reduce_min(255.0 - mean)
        elif prepro_mode == 'SIMPLE_NORM':
            clip_min = -0.5
            clip_max = 0.5
        elif prepro_mode == '0_1_NORM':
            clip_min = 0.0
            clip_max = 1.0
        else:
            clip_min = 0.0
            clip_max = 255.0

        return clip_min, clip_max

    def get_preprocess(self):

        xadv = self.img_org + self.noise_var
        xadv = tf.maximum(xadv, self.img_org - self.epsilon)
        xadv = tf.minimum(xadv, self.img_org + self.epsilon)
        xadv = tf.clip_by_value(xadv, self.clip_min, self.clip_max)

        aa_input = tf.case([(tf.equal(self.aa_init_on, True), lambda: self.img_org)], default=lambda: xadv)

        return aa_input

    def get_kl_robustness(self, temp=0.1):
        num_classes = self.params['Dataset']['num_classes']
        logits_shape = [None, num_classes]

        self.logits_org = tf.placeholder(dtype=tf.float32, shape=logits_shape, name='original_logits')
        self.logits_aa = tf.placeholder(dtype=tf.float32, shape=logits_shape, name='attack_logits')

        q_logits_softmax = tf.nn.softmax(self.logits_org / temp)

        p_logits_softmax = tf.nn.softmax(self.logits_aa / temp)

        kl_div = tf.reduce_sum(p_logits_softmax * tf.log(p_logits_softmax / q_logits_softmax), axis=-1)

        kl_robust = 1.0 / tf.reduce_max(kl_div)

        return kl_robust

    def get_xent_robustness(self):

        xent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.logits_org, logits=self.logits_aa)
        xent_robust = 1.0 / tf.reduce_mean(xent_loss)

        return xent_robust