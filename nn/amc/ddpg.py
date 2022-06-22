import numpy as np
import sys
import tensorflow as tf
from nn import layer
from ops.learning_rate import learning_rate
# from tf_collection.collection import *

from nn.amc.memory import SequentialMemory

"""
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG.py
"""


class Actor():
    def __init__(self, state_var, new_state_var, action_var, params, name=None):

        self.tau = params['DDPG']['tau']
        self.hidden1 = params['DDPG']['hidden1']
        self.hidden2 = params['DDPG']['hidden2']

        self.parameter = params

        self.act_lb = self.parameter['DDPG']['actor_bounds'][0]
        self.act_rb = self.parameter['DDPG']['actor_bounds'][1]

        self.lr_org = params['DDPG']['actor_lr']
        self.lr = self.lr_org

        self.state = state_var  # s_{i}
        self.state_new = new_state_var  # s_{i+1}
        self.state_shape = self.state.get_shape().as_list()[-1]

        self.action = action_var
        self.action_shape = action_var.get_shape().as_list()[-1]

        with tf.variable_scope('Actor'):
            self.online_a = self.inference(state=self.state, scope='online_net', trainable=True)
            self.target_a = self.inference(state=self.state_new, scope='target_net', trainable=False)

        # on_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/online_net')
        self.on_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/online_net')

        # tar_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/target_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def inference(self, state, scope, trainable):
        with tf.variable_scope(scope):
            with tf.variable_scope('fc1'):
                fc1_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                    'bias': {'init': {'type': 'const', 'val': 0.001}, 'lr_mult': 2, 'decay_mult': 0},
                                    'wd': 0.0005, 'loss': 'l2'}

                self.h_a_fc1 = layer.get_fc_layer(x=state, co=self.hidden1, params=fc1_train_params, trainable=trainable)

            with tf.variable_scope('relu1'):
                self.h_a_relu1 = tf.nn.relu(self.h_a_fc1, name="h_a_relu1")

            with tf.variable_scope('fc2'):
                fc2_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                    'bias': {'init': {'type': 'const', 'val': 0.001}, 'lr_mult': 2, 'decay_mult': 0},
                                    'wd': 0.0005, 'loss': 'l2'}

                self.h_a_fc2 = layer.get_fc_layer(x=self.h_a_relu1, co=self.hidden2, params=fc2_train_params, trainable=trainable)

            with tf.variable_scope('relu2'):
                self.h_a_relu2 = tf.nn.relu(self.h_a_fc2, name="h_a_relu2")

            with tf.variable_scope('fc3'):
                fc3_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                    'bias': {'init': {'type': 'const', 'val': 0.001}, 'lr_mult': 2, 'decay_mult': 0},
                                    'wd': 0.0005, 'loss': 'l2'}

                self.h_a_fc3 = layer.get_fc_layer(x=self.h_a_relu2, co=self.action_shape, params=fc3_train_params, trainable=trainable)

        self.action_new = tf.nn.sigmoid(self.h_a_fc3)

        return self.action_new

    def training(self, critic_q):

        # with tf.variable_scope('Policy_Gradients'):
        #     # ys = policy;
        #     # xs = policy's parameters;
        #     # a_grads = the gradients of the policy to get more Q
        #     # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
        #
        #     action_gradients = tf.negative(a_grads)
        #     self.policy_grads = tf.gradients(ys=self.online_a, xs=self.on_params, grad_ys=action_gradients)
        #
        # with tf.variable_scope('Actor_Opt'):
        #     opt = tf.train.AdamOptimizer(learning_rate=self.lr.var)
        #     self.train_op = opt.apply_gradients(zip(self.policy_grads, self.on_params))

        with tf.variable_scope('Actor_train'):
            self.loss = tf.reduce_mean(critic_q)
            self.train_op = tf.train.AdamOptimizer(-self.lr).minimize(self.loss, var_list=self.on_params)

    def hard_update(self):
        self.hard_update_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.on_params)]

    def soft_update(self):
        self.soft_update_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.on_params)]


class Critic():
    def __init__(self, state_var, new_state_var, action_var, online_a, next_action_var, reward_var, terminals, params, name=None):

        self.tau = params['DDPG']['tau']
        self.hidden1 = params['DDPG']['hidden1']
        self.hidden2 = params['DDPG']['hidden2']

        self.gamma = params['DDPG']['critic_gamma']

        self.parameter = params

        self.lr_org = params['DDPG']['critic_lr']
        self.lr = self.lr_org

        self.state = state_var
        self.state_next = new_state_var
        self.state_shape = self.state.get_shape().as_list()[-1]

        self.batch_a = action_var  # From batch actions
        self.online_a = online_a  # From online_a in Actor
        self.action_next = next_action_var  # From target_actor
        self.action_shape = self.batch_a.get_shape().as_list()[-1]

        self.rewards = reward_var

        self.terminals = terminals

        with tf.variable_scope('Critic'):
            self.online_q = self.inference(state=self.state, action=self.batch_a, scope='online_net', trainable=True, reuse=False)
            self.online_qwa = self.inference(state=self.state, action=self.online_a, scope='online_net', trainable=True, reuse=True)
            self.target_q = self.inference(state=self.state_next, action=self.action_next, scope='target_net', trainable=False, reuse=False)

        self.on_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/online_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

    def inference(self, state, action, scope, trainable, reuse):
        with tf.variable_scope(scope, reuse=reuse):

            with tf.variable_scope('fc11'):
                fc11_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                     'bias': {'init': {'type': 'const', 'val': 0.1}, 'lr_mult': 2, 'decay_mult': 0},
                                     'wd': 0.0005, 'loss': 'l2'}

                self.h_s_fc11 = layer.get_fc_layer(x=state, co=self.hidden1, params=fc11_train_params, trainable=trainable)

            with tf.variable_scope('fc12'):
                fc12_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                     'bias': {'init': {'type': 'const', 'val': 0.1}, 'lr_mult': 2, 'decay_mult': 0},
                                     'wd': 0.0005, 'loss': 'l2'}

                self.h_s_fc12 = layer.get_fc_layer(x=action, co=self.hidden1, params=fc12_train_params, trainable=trainable)

            with tf.variable_scope('relu1'):
                self.h_s_relu1 = tf.nn.relu(self.h_s_fc11 + self.h_s_fc12, name="h_s_relu1")

            with tf.variable_scope('fc2'):
                fc2_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                    'bias': {'init': {'type': 'const', 'val': 0.1}, 'lr_mult': 2, 'decay_mult': 0},
                                    'wd': 0.0005, 'loss': 'l2'}

                self.h_s_fc2 = layer.get_fc_layer(x=self.h_s_relu1, co=self.hidden2, params=fc2_train_params, trainable=trainable)

            with tf.variable_scope('relu2'):
                self.h_s_relu2 = tf.nn.relu(self.h_s_fc2, name="h_s_relu2")

            with tf.variable_scope('fc3'):
                fc3_train_params = {'weight': {'init': {'type': 'rand', 'std': 0.01}, 'lr_mult': 1, 'decay_mult': 1},
                                    'bias': {'init': {'type': 'const', 'val': 0.1}, 'lr_mult': 2, 'decay_mult': 0},
                                    'wd': 0.0005, 'loss': 'l2'}

                self.h_s_fc3 = layer.get_fc_layer(x=self.h_s_relu2, co=1, params=fc3_train_params, trainable=trainable)

        return self.h_s_fc3

    def compute_loss(self):
        with tf.variable_scope('TD_error'):
            y_i = self.rewards + self.gamma * self.terminals * self.target_q
            loss = tf.reduce_mean(tf.squared_difference(y_i, self.online_q))

            tf.summary.scalar('Critic loss', loss)

        return loss

    # def actor_gradient(self):
    #     self.a_grads = tf.gradients(self.online_q, self.action)[0]

    def training(self):
        with tf.variable_scope('Critic_train'):
            self.loss = self.compute_loss()
            # self.train_op = train.train_op(self, self.loss, var_list=self.on_params)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.on_params)

    def hard_update(self):
        self.hard_update_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.on_params)]

    def soft_update(self):
        self.soft_update_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.on_params)]


class DDPG(object):
    def __init__(self, state_dims, action_dims, num_prunable, params, dataset):  # state_dims, action_dims, parameters, episode
        self.params = params
        self.warmup = self.params['DDPG']['warmup']
        self.sample_bsize = params['DDPG']['batchsize']

        self.states_shape = [self.sample_bsize, state_dims]
        self.actions_shape = [self.sample_bsize, action_dims]

        self.dataset = dataset

        self.cur_obs = tf.placeholder(name='current_obs', dtype=tf.float32, shape=[state_dims])
        self.episode = tf.placeholder(name='episode', dtype=tf.int32)

        # Parameter configuration for DDPG training
        self.num_to_decay = self.params['DDPG']['lr_decay_eps'] + self.params['DDPG']['warmup']
        self.lr_decay = self.params['DDPG']['lr_decay']
        self.train_iters = self.params['DDPG']['train_iters']

        # Activate Memory
        self.memory_limit = params['DDPG']['memory_capacity'] * num_prunable
        self.Memory = SequentialMemory(limit=self.memory_limit, window_length=1)

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

        # Define variables for saving batch_s, new_batch_s and batch_a
        self.batch_s = tf.get_variable(name='batch_s',
                                       shape=self.states_shape,
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.next_batch_s = tf.get_variable(name='next_batch_s',
                                            shape=self.states_shape,
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer(),
                                            trainable=False,
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.batch_a = tf.get_variable(name='batch_a',
                                       shape=self.actions_shape,
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.batch_r = tf.get_variable(name='batch_r',
                                       shape=self.actions_shape,
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.batch_t = tf.get_variable(name='batch_t',
                                       shape=self.actions_shape,
                                       dtype=tf.float32,
                                       initializer=tf.ones_initializer(),
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        # Define arrays for saving the dataset from Memory
        self.batch_state = tf.placeholder(name='batch_s', shape=self.states_shape, dtype=tf.float32)
        self.new_batch_state = tf.placeholder(name='batch_s_', shape=self.states_shape, dtype=tf.float32)
        self.batch_action = tf.placeholder(name='batch_a', shape=self.actions_shape, dtype=tf.float32)
        self.batch_reward = tf.placeholder(name='batch_r', shape=self.actions_shape, dtype=tf.float32)
        self.batch_terminals = tf.placeholder(name='batch_t', shape=self.actions_shape, dtype=tf.float32)


        # Build up Actor and Critic
        self.Actor = Actor(self.batch_s, self.next_batch_s, self.batch_a, params, name='Actor')
        self.Actor.hard_update()
        self.Actor.soft_update()
        self.action_tar = self.Actor.target_a  # Prediction from target_actor net

        self.Critic = Critic(self.batch_s, self.next_batch_s, self.batch_a, self.Actor.online_a,
                             self.action_tar, self.batch_r, self.batch_t, params, name='Critic')
        self.Critic.hard_update()
        self.Critic.soft_update()
        self.Critic.training()
        # self.Critic.actor_gradient()

        self.Actor.training(self.Critic.online_qwa)

        self.state_update()
        self.select_actions()
        self.batch_update()

        self.batch_a_update = self.batch_a.assign(self.Actor.online_a)

    def state_update(self):
        obs_reshape = tf.expand_dims(self.cur_obs, axis=0)  # [[layer, ci, co, s, k, reduced, rest, a_{t-1}]]
        obs_replicated = tf.cast(tf.tile(obs_reshape, multiples=[self.sample_bsize, 1]), dtype=tf.float32)

        self.replicate_s_op = self.batch_s.assign(obs_replicated)

    def select_actions(self):  # (Step 1.)
        lbound = self.params['DDPG']['actor_bounds'][0]
        rbound = self.params['DDPG']['actor_bounds'][1]
        delta_decay = self.params['DDPG']['delta_decay']
        sigma = self.params['DDPG']['sigma']
        target_sparsity = self.params['Prune_config']['target_sparsity']

        def random_action():
            # action = tf.clip_by_value(tf.abs(tf.random_normal(shape=[1], mean=0.0, stddev=0.25)), 0.0, 1.0)
            action = tf.random_uniform(shape=[1], minval=0.0, maxval=1.0, dtype=tf.float32)
            return action

        def compute_action():
            action_deter = tf.squeeze(self.Actor.online_a)[0]  # single deterministic action
            episode = tf.cast(self.episode, dtype=tf.float32)
            sigma_decay = sigma * (delta_decay ** (episode - self.warmup))
            # real_action = tf.abs(tf.truncated_normal(shape=[1], mean=action_deter, stddev=sigma_decay, dtype=tf.float32))
            # real_action = tf.clip_by_value(real_action, 0.0, 1.0)
            # inscale_action = real_action * (rbound - lbound) + lbound  # Rescale action from [0, 1] into [lbound, rbound]

            action = tf.clip_by_value(
                tf.abs(tf.truncated_normal(shape=[1], mean=action_deter, stddev=sigma_decay, dtype=tf.float32)),
                0.0, 1.0)

            # In the environment, action wil be rescaled into valid range, here we only obtain the action in range [0.0, 1.0]

            return action

        # single action from normally truncated sample
        self.get_action_op = tf.case([(tf.greater(self.episode, self.warmup), lambda: compute_action())],
                                     default=lambda: random_action())

    def save_to_memory(self, s_t, a_t, r_t, done):  # (Step 2.)
        self.Memory.append(s_t, a_t, r_t, done)

    # Sample batch from Memory
    def sample_batch(self):
        # Fetch batch dataset from Memory
        batch_state, batch_action, batch_reward, new_batch_state, batch_terminal \
            = self.Memory.sample_and_split(batch_size=self.sample_bsize)  # (Step 3.)

        return batch_state, batch_action, batch_reward, new_batch_state, batch_terminal

    # Update batch variables
    def batch_update(self):  # (Step 3.)
        # normalize the reward
        batch_mean_reward = tf.reduce_mean(self.batch_reward)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)

        batch_reward = self.batch_reward - self.moving_average

        # Assign operations for all inputs variables
        state_assign = self.batch_s.assign(self.batch_state)
        new_state_assign = self.next_batch_s.assign(self.new_batch_state)
        action_assign = self.batch_a.assign(self.batch_action)
        reward_assign = self.batch_r.assign(batch_reward)
        terminal_assign = self.batch_t.assign(self.batch_terminals)

        self.batch_update_op = [state_assign, new_state_assign, action_assign, reward_assign, terminal_assign]

    def summary_merge(self, tb_writer, step):
        tb_merged = tf.summary.merge_all()
        tb_writer.add_summary(tb_merged, step)

    def _progress_bar(self, step):
        num_steps = self.train_iters
        bar_len = num_steps
        bars = int((float(step) / num_steps) * bar_len)
        sys.stdout.write('\r (%d/%d)' % (step, num_steps))
        for b in range(bars):
            sys.stdout.write('|')
        sys.stdout.flush()

    def update_policy(self, sess, episode, step):

        total_steps = self.train_iters
        train_step = step + 1
        sum_c_loss = 0.0
        sum_a_loss = 0.0

        # Update learning rate for Actor & Critic with lr_decay
        if episode >= self.num_to_decay:
            # self.Actor.lr = self.Actor.lr_org * (self.lr_decay ** (episode - self.num_to_decay))
            # self.Critic.lr = self.Critic.lr_org * (self.lr_decay ** (episode - self.num_to_decay))
            self.Actor.lr = self.Actor.lr * self.lr_decay
            self.Critic.lr = self.Critic.lr * self.lr_decay

        # Hard update at begin
        if episode == self.warmup:
            sess.run([self.Actor.hard_update_op, self.Critic.hard_update_op])

        # Sample batch from reply buffer
        s, a, r, s_, t = self.sample_batch()
        sess.run(self.batch_update_op, feed_dict={self.batch_state: s,
                                                  self.batch_action: a,
                                                  self.batch_reward: r,
                                                  self.new_batch_state: s_,
                                                  self.batch_terminals: t})

        # Critic Optimization
        _, critic_loss = sess.run([self.Critic.train_op, self.Critic.loss])
        sum_c_loss += critic_loss

        # Actor Optimization
        _, actor_loss = sess.run([self.Actor.train_op, self.Actor.loss])
        sum_a_loss += actor_loss

        # Soft update
        sess.run([self.Actor.soft_update_op, self.Critic.soft_update_op])

        # Print training results
        if train_step == 1:
            print('Agent Training:')
        self._progress_bar(train_step)

        if train_step == total_steps:
            # mean_loss = sum_loss / total_steps
            print((' Sum Critic loss: %.5f' % sum_c_loss) + (' | Sum Actor loss: %.5f' % sum_a_loss))

        # Add Summary to Tensorboard
        # self.summary_merge(sess, tb_writer, episode)
