import tensorflow as tf
import numpy as np
from nn.attacks.attack_base import Attack_Base
import math

'''
    ###################################
    ######   White-Box attacks   ######
    ###################################
    
    Projected Gradient Descent (PGD) Attack - 'multiple classes and batch images' implementation.
    See https://arxiv.org/abs/1706.06083 for details.

    Reference: https://github.com/MadryLab/cifar10_challenge

    Here only Linf-loss & Cross_Entropy loss are used for this implementation.
'''


class PGD(Attack_Base):
    def __init__(self, img, lab, params, ds_switch):

        Attack_Base.__init__(self, img, lab, params, ds_switch)

        self.stepsize = params['Attack_config']['stepsize']

    def attack_op(self, model, loss):
        self.model = model
        self.logits = model.logits
        self.total_loss = loss

        update_noise_op = self.pgd()

        return update_noise_op

    def reset_attack_op(self):
        reset_list = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if var.name.find('noise') != -1]
        reset_op = tf.variables_initializer(reset_list)
        return reset_op

    def pgd(self):
        if self.params['Attack_config']['pgd_loss'] == 'linf':
            logits = self.logits
            pred_softmax = tf.nn.softmax(logits)
            label_mask = tf.one_hot(self.lab_var, self.params['Dataset']['num_classes'], on_value=1.0, off_value=0.0, dtype=tf.float32)
            correct_logits = tf.reduce_sum(label_mask * pred_softmax, axis=1)
            wrong_logits = tf.reduce_max((1-label_mask) * pred_softmax - 1e8 * label_mask, axis=1)
            # Linf-Loss
            loss = - tf.nn.relu(correct_logits - wrong_logits + 50)

        elif self.params['Attack_config']['pgd_loss'] == 'xent':
            # Cross_Entropy Loss
            loss = self.total_loss
        else:
            raise NotImplementedError('Please check the name of "pgd_loss"')

        loss_grad, = tf.gradients(loss, self.img)
        sign_grad = tf.sign(loss_grad)
        new_noise = self.noise_var + self.stepsize * sign_grad
        new_noise = tf.clip_by_value(new_noise, -self.epsilon, self.epsilon)

        update_noise_op = self.noise_var.assign(new_noise)

        return update_noise_op
