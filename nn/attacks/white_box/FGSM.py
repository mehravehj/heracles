import tensorflow as tf
import numpy as np
from nn.attacks.attack_base import Attack_Base
import math

'''
    ###################################
    ######   White-Box attacks   ######
    ###################################
    
    Fast Gradient Sign Method (FGSM)

    See https://arxiv.org/pdf/1412.6572.pdf for details.
'''


class FGSM(Attack_Base):
    def __init__(self, img, lab, params, ds_switch):

        Attack_Base.__init__(self, img, lab, params, ds_switch)

    def attack_op(self, model, loss):
        self.model = model
        self.logits = self.model.logits
        self.total_loss = loss

        update_noise_op = self.fgsm()

        return update_noise_op

    def attack_eval_op(self):
        aa_eval_op = self.pgd_eval()
        return aa_eval_op

    def reset_attack_op(self):
        reset_list = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if var.name.find('noise') != -1]
        reset_op = tf.variables_initializer(reset_list)
        return reset_op

    def fgsm(self):
        """ Non-Target-iterative-FGSM Methods.
            See https://arxiv.org/pdf/1412.6572.pdf for details.
        """
        loss = self.total_loss
        loss_grad, = tf.gradients(loss, self.img)
        sign_grad = tf.sign(loss_grad)

        if self.target_on:
            new_noise = self.noise_var - self.stepsize * sign_grad
        else:
            new_noise = self.noise_var + self.stepsize * sign_grad

        new_noise = tf.clip_by_value(new_noise, -self.epsilon, self.epsilon)

        update_noise_op = self.noise_var.assign(new_noise)

        return update_noise_op

    def pgd_eval(self):
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
        new_noise = self.noise_var + self.eval_stepsize * sign_grad
        new_noise = tf.clip_by_value(new_noise, -self.eval_eps, self.eval_eps)

        update_noise_op = self.noise_var.assign(new_noise)

        return update_noise_op