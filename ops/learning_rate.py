import tensorflow as tf
import math


class learning_rate():
    def __init__(self, params):
        self.params = params
        self._base_lr = self.params['Train_config']['lr']
        self._base_momentum = self.params['Train_config']['momentum']
        self._police = self.params['Train_config']['lr_police']
        self._lr_gamma = self.params['Train_config']['lr_gamma']
        self._lr_decay = self.params['Train_config']['lr_decay']
        self._power = self.params['Train_config']['power']
        self._epoch_steps = self.params['Dataset']['num_train_samples'] / self.params['Meta']['batchsize']
        self._step_size = int(self.params['Train_config']['lr_step_size'] * self._epoch_steps)
        self._max_step = int(self.params['Meta']['epochs'] * self._epoch_steps)

        self._value = self._base_lr

        self.lr_var = tf.get_variable(name='learning_rate', shape=[], trainable=False,
                                      initializer=tf.constant_initializer(self._base_lr),
                                      collections=[tf.GraphKeys.LOCAL_VARIABLES])

        self.momentum = tf.get_variable(name='momentum', shape=[], trainable=False,
                                        initializer=tf.constant_initializer(self._base_momentum),
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])

        self.val_temp = tf.placeholder(dtype=tf.float32, shape=[])
        self.update_op = tf.assign(self.lr_var, self.val_temp)

    def set_learning_rate(self, step):
        if self._police == 'fixed':
            self._value = self._base_lr
        elif self._police == 'step':
            self._value = self._base_lr * pow(self._lr_gamma, math.floor(step / self._step_size))
        elif self._police == 'decay':
            epoch = math.floor(step / self._step_size)
            self._value = self._base_lr * pow(1/(1 + self._lr_decay * epoch), epoch)
        elif self._police == 'exp':
            self._value = self._base_lr * pow(0.5, step // self._epoch_steps)
        elif self._police == 'inv':
            self._value = self._base_lr * pow(1 + self._lr_gamma * step, -self._power)
        elif self._police == 'poly':
            self._value = self._base_lr * pow(1 - step / self._max_step, self._power)
        elif self._police == 'cyclic':
            period = self._step_size
            turn_point = self._step_size / 3.0
            step_cycle = step % period
            if step_cycle > turn_point:
                cycle_lr = self._base_lr * (1.0 - (step_cycle - turn_point) / (2.0/3.0 * period))
            else:
                cycle_lr = self._base_lr * step_cycle / turn_point
            self._value = max(cycle_lr, 1e-8) * pow(0.99995, step // period * period)
        elif self._police == 'cosine':
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self._max_step))
            decayed = (1 - 1e-6) * cosine_decay + 1e-6
            self._value = self._base_lr * decayed
        elif self._police == 'schedule':
            schedules = self.params['Train_config']['lr_schedule']
            for ep in schedules:
                ep_steps = ep * self._epoch_steps
                if step >= ep_steps:
                    self._value *= self._lr_decay
        else:
            raise NameError('learning rate police is incorrect!')
