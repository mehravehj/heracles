import json
import os

import numpy as np
import torch
import tensorflow as tf
import argparse
from dataset import dataset
from utils.evaluation import eval_classification
from utils.evaluation import BN_PERCEPTION_MODE, BN_TRAIN_MODE

from utils.model_zoo import fetch_model
from utils.tf_gpu_config import tf_gpu_config
from utils import torch_2_tf

parser = argparse.ArgumentParser(description='Model Reload to TF')
parser.add_argument('--arch', type=str, default='vgg16_bn', choices={'vgg16_bn', 'resnet18', 'wrn_28_4', 'resnet50'}, help='model for pruning')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices={'CIFAR10', 'SVHN', 'imagenet'})
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--test_acc', type=bool, default=False, help='test model performance on natural test set')
parser.add_argument('--seed', type=int, default=1234, help='random seed for reproduce')

args = parser.parse_args()

torch_model = f"./Pretrain-Models/torch-models/{args.arch}_pretrain_{args.dataset.lower()}.pth.tar"
output_name = f"{args.arch}_pretrain_{args.dataset.lower()}"
out_dir = f"./Pretrain-Models/tf-models"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

cwd = os.getcwd()

config_file = f'config_{args.arch}_{args.dataset.lower()}.json'
session_config = json.load(open(os.path.join(cwd, 'configs', config_file)))
print('\nLoad configuration file from {}\n'.format(config_file))

# out_dir = os.path.join(cwd, session_config['Meta']['output_folder'])
if not os.path.exists(out_dir):
    new_dir = out_dir.split('/')
    out_dir = '/'
    for f in new_dir:
        out_dir = os.path.join(out_dir, f)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

# Specify used GPU
os.environ["CUDA_VISIBLE_DEVICES"] = session_config['Gpu_config']['cuda_visible_devices']

def main(_):
    model_name, NN_model = fetch_model(session_config)

    restore_path = torch_model
    # restore_path = os.path.join(session_config['Meta']['input_folder'], session_config['Meta']['restore_model'])

    torch_var_list = []  # variables that exit in torch model and also match tensorflow
    use_torch_model = False
    if restore_path.find('.pth.tar') != -1:
        net_key = 'state_dict'
    elif restore_path.find('.pt') != -1:
        net_key = 'net'
    else:
        raise NameError(f'{restore_path} is not supported. Only support ".pth.tar" and ".pt"')

    print('Using PyTorch checkpoint......')
    use_torch_model = True
    checkpoint = torch.load(restore_path, map_location=torch.device('cpu'))
    torch_dict = checkpoint[net_key]

    tf_dict = {}
    for k in torch_dict.keys():
        # filter out subblock since it is not included in wrn284 model
        if model_name == 'WRN_28_4' and k.find('sub_block1') != -1:
            continue
        else:
            tf_v, is_load = torch_2_tf.match_var(k, torch_dict, model_name)

        if is_load:
            tf_dict.update(tf_v)
            torch_var_list.append(list(tf_v.keys())[0])

    with tf.Graph().as_default():
        gpu_config = tf_gpu_config(session_config)

        is_training = tf.placeholder_with_default(1, (), name='is_training')

        ds = dataset.Dataset(session_config)
        img, lab = ds.get_batch()

        ''' Build Model '''
        with tf.variable_scope(model_name):
            model = NN_model(params=session_config, ds_switch=ds.ds_switch)
            model.inference(img=img)
            model.training(lab=lab)
            model.evaluation(lab=lab)
            # model.explanation(lab=lab)
            print('Successfully Build Model!')

        var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

        if use_torch_model:
            var_restore_ops = []
            var_ph_dict = {}
            for var in var_list:
                if var.name in torch_var_list:
                    v_ph_name = 'PH_'+var.name[:-2]
                    var_ph = tf.placeholder(name=v_ph_name, shape=var.shape, dtype=var.dtype)
                    var_ph_dict.update({v_ph_name: var_ph})
                    var_assign_op = tf.assign(var, var_ph)
                    var_restore_ops.append(var_assign_op)

        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

        with tf.Session(config=gpu_config) as sess:
            sess.run(global_init)
            sess.run(local_init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if session_config['Prune_config']['prune_reg'] != '':
                mask_vars = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if var.name.find('mask') != -1]
                var_list.extend(mask_vars)

            if use_torch_model:
                feed_dict = {}
                for tf_k in tf_dict.keys():
                    tf_to_vname = tf_dict[tf_k][0]
                    tf_v = tf_dict[tf_k][1]
                    v_ph_n = 'PH_'+tf_k[:-2]
                    v_ph = var_ph_dict[v_ph_n]
                    feed_dict.update({v_ph: tf_v})

                    print('> {:55}  ==>  {}'.format(tf_k, tf_to_vname))

                sess.run(var_restore_ops, feed_dict=feed_dict)

                saver = tf.train.Saver(var_list=var_list)
            else:
                if session_config['Meta']['restore_model'] != '':
                    saver = tf.train.Saver(var_list=var_list)
                    saver.restore(sess=sess, save_path=restore_path)

            if args.test_acc:
                print('BN Perception Mode:')
                te_acc_1, _, _, _ = eval_classification(sess, model.eval_op, session_config, ds.TEST_SET, ds, is_training, bn_mode=BN_PERCEPTION_MODE, print_acc=False)
                print('Accuracy on Testset: {}%'.format(te_acc_1))

            checkpoint_file = os.path.join(out_dir, output_name + '.ckpt')

            saver.save(sess, checkpoint_file)

            coord.request_stop()
            coord.join(threads)

        del sess
        del model

        print('Progress done!')


if __name__ == '__main__':
    tf.app.run()
