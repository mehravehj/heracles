import os

import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from copy import deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta

from dataset import dataset
from utils.evaluation import evaluate_attack
from utils.parse_config import parse_config
from utils.model_zoo import fetch_model
from utils.tf_gpu_config import tf_gpu_config

from nn.attacks.aa_utils import fetch_attack

from nn.amc.ddpg import DDPG
from nn.amc.robust_global_prune_env import PruningEnv

parser = argparse.ArgumentParser(description='Heracles Robust Pruning')
parser.add_argument('--arch', type=str, default='vgg16_bn', choices={'vgg16_bn', 'resnet18', 'wrn_28_4', 'resnet50'}, help='model for pruning')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices={'CIFAR10', 'SVHN', 'imagenet'})
parser.add_argument('--output_name', type=str, default='', help='output dir for saving snapshots')
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--batchsize', type=int, default=128, help='batchsize for performance evaluation')
parser.add_argument('--target_sparsity', type=float, default=0.5, help='target prune rate')
parser.add_argument('--prune_reg', type=str, default='channel', help='prune regularity')
parser.add_argument('--prune_1st', type=bool, default=False, help='prune first layer')
parser.add_argument('--actor_lbound', type=float, default=0.1, help='actor lower bound')
parser.add_argument('--actor_ubound', type=float, default=1.0, help='actor upper bound')
parser.add_argument('--seed', type=int, default=1234, help='random seed for reproduce')

args = parser.parse_args()

session_config = parse_config(args)

# Specify used GPU
os.environ["CUDA_VISIBLE_DEVICES"] = session_config['Gpu_config']['cuda_visible_devices']


def main(_):
    model_name, NN_model = fetch_model(session_config)

    with tf.Graph().as_default():
        gpu_config = tf_gpu_config(session_config)

        # Set random seed
        if args.seed != 0:
            random.seed(args.seed)
            os.environ['PYTHONHASHSEED'] = str(args.seed)
            np.random.seed(args.seed)
            tf.set_random_seed(args.seed)

        is_training = tf.placeholder_with_default(1, (), name='is_training')

        ds = dataset.Dataset(session_config)
        img, lab = ds.get_batch()

        ''' Build attacks '''
        with tf.variable_scope('Attack'):
            Attack = fetch_attack(session_config)
            attack = Attack(img=img, lab=lab, params=session_config, ds_switch=ds.ds_switch)
            aa_img = attack.img
            aa_img_org = attack.img_org
            aa_lab = attack.lab_var
            aa_lab_org = attack.lab_org
            print('Successfully Initialized Attack Class')

        ''' Build Model '''
        with tf.variable_scope(model_name):
            # model = NN_model(params=session_config, ds_switch=ds.ds_switch, input=aa_img, bng_input=aa_img_org)
            model = NN_model(params=session_config, ds_switch=ds.ds_switch)
            model.inference(img=aa_img)

            # Prepare learning for fine-tuning
            model.lr._base_lr = session_config['Prune_config']['score_optim_lr']
            model.lr._step_size = int(10 * model.lr._epoch_steps)

            model.training(lab=aa_lab_org)
            model.evaluation(lab=aa_lab_org)
            model.hw_struct()

            print('Successfully Build Model!')

        ''' Build attacks Ops'''
        get_aa_inputs = attack.get_adv_input()
        attack_op = attack.attack_op(model=model, loss=model.total_loss)
        aa_reset_op = attack.reset_attack_op()

        aa_ops = {'get_inputs': get_aa_inputs,
                  'get_attack': attack_op,
                  'reset_attack': aa_reset_op}

        '''Build Prune Environment'''
        env = PruningEnv(model, model.hw_structs, ds.VALID_SET, ds, session_config, is_training, adv_attack=attack, aa_ops=aa_ops)

        '''Build DRL agent'''
        state_dims = len(env.layer_embedding[0])
        action_dims = 1  # All layer has only one action which means compression ratio
        num_prunable = env.n_layers - 1
        agent = DDPG(state_dims, action_dims, num_prunable, session_config, ds)

        print('Pruning Agent is done!')

        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

        with tf.Session(config=gpu_config) as sess:
            sess.run(global_init)
            sess.run(local_init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            restore_path = os.path.join(os.getcwd(), session_config['Meta']['input_folder'], session_config['Meta']['restore_model'])

            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                        if var.name.find(model_name) != -1 and var.name.find('p_') == -1]
            p_mask_list = list(env.mask_dict.values())
            p_rate_list = list(env.rate_dict.values())

            print(f'==> Loading from {restore_path}')
            reader = pywrap_tensorflow.NewCheckpointReader(restore_path)  # tf.train.NewCheckpointReader
            checkpoint_vars = reader.get_variable_to_shape_map()
            restore_var_list = []
            for var in var_list:
                var_name = var.name[:-2]
                if var_name not in checkpoint_vars:
                    print('Not Found Variable: {} !'.format(var_name))
                else:
                    print('    Found Variable: {}.'.format(var_name))
                    restore_var_list.append(var)

            save_var_list = var_list + p_mask_list + p_rate_list

            saver_restore = tf.train.Saver(var_list=restore_var_list)
            saver_restore.restore(sess=sess, save_path=restore_path)
            print('Model restored from {}'.format(restore_path))

            sess.run(env.mask_init_op)
            print('Prune Masks Initialized!')

            evaluate_attack(sess, model, attack, aa_ops, ds, run_trainset=False,
                            run_tr_mode=False, params=session_config, is_training=is_training)


            """ ---------------------------- RL Model Pruning ------------------------------------- """

            print("\n================= Start Heracles Pruning =================\n")

            '''Run Pruning'''
            episode = 1
            episode_reward = 0.0
            observation = None
            Template = []

            start_time = datetime.now()

            while episode <= session_config['DDPG']['train_episodes'] + session_config['DDPG']['warmup']:

                if observation is None:
                    print('Episode: %d' % episode)
                    observation = deepcopy(env.reset())

                # Agent start to pick action
                if episode > session_config['DDPG']['warmup']:
                    sess.run(agent.replicate_s_op, feed_dict={agent.cur_obs: observation})

                action = sess.run(agent.get_action_op, feed_dict={agent.episode: episode, agent.cur_obs: observation})

                # Get observation and reward from Environment with action from DDPG
                observation_2, reward, terminal = env.step(sess, action, episode, is_training, save_var_list)

                # Save data into Template
                Template.append([reward, deepcopy(observation), action, terminal])

                episode_reward += reward
                observation = deepcopy(observation_2)

                if terminal:
                    # sess.run(env.rate_reset_op)
                    # print('Prune Rates Recovered!')

                    if episode_reward > session_config['DDPG']['min_reward'] \
                            or episode >= session_config['DDPG']['warmup']:
                        print('\x1b[6;30;44m' + ' Reward after pruning: ' + '\x1b[0m', '%.3f\n' % episode_reward)

                        final_reward = Template[-1][0]

                        for r_t, s_t, a_t, done in Template:
                            agent.save_to_memory(s_t, a_t, final_reward, done)

                        episode += 1

                    if episode >= session_config['DDPG']['warmup']:
                        if episode == session_config['DDPG']['warmup']:
                            init_time = datetime.now() - start_time
                            print('******** Time for Initialization: ', init_time)

                            print('\n**************************************')
                            print('           Start Training             ')
                            print('**************************************')

                        for step in range(session_config['DDPG']['train_iters']):
                            agent.update_policy(sess, episode, step)

                    observation = None
                    episode_reward = 0.0
                    Template = []

                    sess.run(env.mask_init_op)
                    print('Prune Masks Re-Initialized!')

            best_reward = env.best_reward
            best_strategy = env.best_strategy
            layer_names = list(env.l_weight_dict.keys())

            end_time = datetime.now()

            prune_time = relativedelta(end_time, start_time)

            try:
                log = open(os.path.join(session_config['Meta']['output_folder'], 'Episode_wise_Rewards.csv'), 'a')
            except:
                log = open(os.path.join(session_config['Meta']['output_folder'], 'Episode_wise_Rewards.csv'), 'w')

            time_print = '\nTotal time use for pure pruning: {h}h:{m}m:{s}s.'.format(h=prune_time.hours,
                                                                                     m=prune_time.minutes,
                                                                                     s=prune_time.seconds)
            log.write(time_print)
            log.close()

            print('******** Whole Model Pruning without Fine Tuning: ', prune_time)

            print('\nBest reweard after pruning: %.3f' % best_reward)
            print('\nBest Compression Strategy: ')
            for i, name in enumerate(layer_names):
                print(name + ': ' + str(best_strategy[i]))
            print('\n')

            """----------------------------- Restoring the best strategy -------------------------------------------"""
            print('\x1b[6;30;42m' + 'Pruning Completed \x1b[0m', end='\n')

            print("Original Estimates:   {}".format(env.original_ops))

            rest_flops = env.best_estimate
            print("Compressed Estimates: {}".format(rest_flops))

            coord.request_stop()
            coord.join(threads)

        del sess
        del model

        print('Progress done!')


if __name__ == '__main__':
    tf.app.run()
