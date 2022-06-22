from __future__ import print_function
from collections import OrderedDict
import copy
import sys
import os
import tensorflow as tf
import numpy as np

from utils import training
from utils.evaluation import eval_classification_attack as evaluate
from nn.amc.count_flops import get_flops
from nn.amc.reward import acc_estimates_reward

from utils.evaluation import BN_TRAIN_MODE


class PruningEnv():
    """
    Environment for Channel Pruning
    """
    def __init__(self, model, model_hf, ds_switch, dataset, params, is_training, adv_attack, aa_ops):
        self.params = params

        self.max_sparsity_ratio = params['Prune_config']['max_sparsity']
        self.l_bound = params['DDPG']['actor_bounds']

        self.prune_reg = self.params['Prune_config']['prune_reg']
        self.prune_1st = self.params['Prune_config']['prune_1st']
        self.sparsity_mode = self.params['Prune_config']['sparsity_mode']
        self.prune_criterion = self.params['Prune_config']['prune_criterion']
        self.count_buffer = self.params['Prune_config']['count_buffer']

        self.params['Meta']['epochs'] = self.params['Prune_config']['score_optim_epochs']  # Use score optimize epochs in searching

        self.model = model
        self.model_hf = model_hf
        self.model_nn_org = self.model_hf.nn
        self.save_original_estimates()
        self.ds_switch = ds_switch
        self.dataset = dataset
        self.is_training = is_training

        self.num_batch = 1

        self.init_ind = 1 if self.prune_reg == 'channel' and (not self.prune_1st) else 0  # Current Conv. layer index to prune
        self.cur_ind = self.init_ind

        self.org_estimates, self.org_estimates_list, self.buffer_list, self.original_channels = get_flops(self.model_hf.nn, self.count_buffer)
        self.shortcut_names = list()
        # self.preserve_original_model()

        # Initialize attack stuffs
        self.attack = adv_attack
        self.kl_robust_op = adv_attack.kl_robust
        self.xent_robust_op = adv_attack.xent_robust
        self.aa_ops = aa_ops

        # Build RL agent data states
        self._build_index()
        self._build_state_embedding()

        self.target_rate = self.get_target_rate()

        # Build reset ops
        self._reset_all_masks()  # Reset all masks as abs(weights)
        self._reset_all_rates()

        self.layer_ind = tf.placeholder(name='layer_ind', dtype=tf.int32)
        self.action = tf.placeholder(name='env_action', dtype=tf.float32)
        self.target_action = tf.placeholder(name='target_action', dtype=tf.float32)

        # real_action: used to save the current real action after channel rounding and layer pruning
        self.real_action = tf.identity(self.action)

        # action_list: used to save all rounded actions for final global pruning
        self.action_list = tf.placeholder(name='action_list', shape=(len(self.l_weight_dict),), dtype=tf.float32)

        self._action_wall()
        self._global_pruning()
        self.best_strategy_dict = self.strategy_dict
        self.best_strategy = None
        self._apply_best_prune()

        self.expected_computation = self.max_sparsity_ratio * self.org_estimates

        self.best_reward = -np.inf
        self.best_acc = 0.0
        self.best_estimate = 0.0

        self.saver = tf.train.Saver()

    # Parse these functions estimate
    def save_original_estimates(self):

        hf_estimate_dict = self.model_hf.start_estimate(self.count_buffer)
        self.original_ops = hf_estimate_dict["total_ops"]
        self.original_layer_wise_flops = hf_estimate_dict["layer_ops"]

    def estimates_list(self, layer_name=""):

        layer_idx = 0

        for idx, l_name in enumerate(self.l_weight_dict.keys()):
            if l_name == layer_name:
                layer_idx = idx
                break

        def _flops_list(self):
            flops_list = []  # FLOPs of all conv. and dense layers
            buffer_list = [0]  # Buffer flops after each conv. or dense layer

            hf_estimate_dict = self.model_hf.start_estimate(self.count_buffer)
            if layer_name:
                return None, hf_estimate_dict["layer_ops"][layer_idx], None
            else:
                for i in range(len(self.model_hf.nn)):
                    cur_flops = hf_estimate_dict["layer_ops"][i]
                    flops_list.append(cur_flops)
                    buffer_list.append(0)

                return hf_estimate_dict["total_ops"], flops_list, buffer_list[:-1]   # From conv1 to fc6 (20 layers)

        total_estimates, estimate_list, estimate_buffer_list = _flops_list(self)

        return total_estimates, estimate_list, estimate_buffer_list

    # Compute current model flops

    def preserve_original_model(self):
        self.original_channels = []
        for i in range(len(self.model_hf.nn)):
            l_type = self.model_hf.nn[i]['type']
            if l_type == 'CONV' or l_type == 'FC':
                self.original_channels.append([self.model_hf.nn[i]['Nif'],  self.model_hf.nn[i]['Nof']])

    def _cur_estimates(self):
        flops = 0
        for i in range(self.n_layers):
            ratio_c_in, ratio_c_out = self.strategy_dict[i]
            l_flops = self.org_estimates_list[i] * ratio_c_in * ratio_c_out  # layer_flops = (k*k*cin*cout*h*w)*r_cin*r_cout
            l_buffer_flops = self.buffer_list[i] * ratio_c_in

            flops += l_flops + l_buffer_flops

        flops += sum(self.org_estimates_list[self.n_layers:] + self.buffer_list[self.n_layers:])
        # current_estimates, cur_estimates_list, cur_buffer_list = self.estimates_list()
        return flops

    def _cur_estimates_reduced(self):

        _, layer_wise_estimates, _= self.estimates_list()

        reduced = 0
        for i in range(self.backbone_ids[self.cur_ind]):
            if self.prune_reg == 'kernel' or self.prune_reg == 'weight':
                ratio = self.strategy_dict[i][0]
                l_flops = self.org_estimates_list[i] * (1 - ratio)
                l_buffer_flops = self.buffer_list[i] + (1 - ratio)
                reduced += l_flops + l_buffer_flops
            else:
                ratio_c_in, ratio_c_out = self.strategy_dict[i]
                l_flops = self.org_estimates_list[i] * (1 - ratio_c_in * ratio_c_out)
                l_buffer_flops = self.buffer_list[i] * (1 - ratio_c_in)
                reduced += l_flops + l_buffer_flops

        return reduced

    def _build_index(self):  # Get all prunable layers ans feature maps
        # Only conv2 - fc6 should be pruned, but in order to make shape of all layers suitable to the neighbor,
        # all layers should be considered
        prune_conv, prune_fc = self.params["Prune_config"]['prune_conv'], self.params["Prune_config"]['prune_fc']

        if self.prune_reg == 'channel' or self.prune_reg == 'filter':
            layer_vars = [var for var in tf.get_collection(tf.GraphKeys.WEIGHTS) if var.name.find('weight') != -1
                          and var.name.find('shortcut') == -1]
            feature_vars = [var for var in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if var.name.find('f_map') != -1
                            and var.name.find('shortcut') == -1
                            and var.name.find('/inference/') != -1]  # For the case of TRADES train

        else:
            layer_vars = [var for var in tf.get_collection(tf.GraphKeys.WEIGHTS) if var.name.find('weight') != -1]
            feature_vars = [var for var in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if var.name.find('f_map') != -1
                            and var.name.find('/inference/') != -1]

        all_layer_vars = [var for var in tf.get_collection(tf.GraphKeys.WEIGHTS) if var.name.find('weight') != -1]

        mask_vars = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if var.name.find('p_mask') != -1]
        rate_vars = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if var.name.find('p_rate') != -1]

        self.backbone_ids = []
        l_id = 0
        for i, l in enumerate(self.model_hf.nn):
            if layer_vars[l_id].name == l['name']:
                self.backbone_ids.append(i)
                l_id += 1

        # Generate input feature map for each layer
        pre_feature_vars = [self.model.inputs]
        all_acts = [act for act in tf.get_collection(tf.GraphKeys.ACTIVATIONS) if act.name.find('shortcut') == -1]
        l_i = 1
        for f_i, fmp in enumerate(all_acts):
            l = layer_vars[l_i]
            l_name = l.name.split('/')[-2]
            pre_fmp = all_acts[f_i - 1]
            if fmp.name.find(l_name) != -1 and pre_fmp.shape[-1]._value == l.shape[-2]._value:
                pre_feature_vars.append(pre_fmp)
                if l_i < len(layer_vars) - 1:
                    l_i += 1
                else:
                    break

        if prune_conv and prune_fc:
            layers = layer_vars  # for VGG16: conv1_1 - fc2
            feature_maps = feature_vars  # for ResNet20: 19 i.e. conv1_1 - before last fc layer
            pre_feature_maps = pre_feature_vars
            masks = mask_vars  # Conv1 is also considered for fitting the filter pruning
            rates = rate_vars
        else:
            if prune_conv:
                l_idxes = [i for i, l in enumerate(layer_vars) if l.name.find('conv') != -1]
            elif prune_fc:
                l_idxes = [i for i, l in enumerate(layer_vars) if l.name.find('fc') != -1]
            else:
                raise NameError('No layer are pruned, please check your prune_config!')

            layers = [l for i, l in enumerate(layer_vars) if i in l_idxes]
            feature_maps = [fm for i, fm in enumerate(feature_vars) if i in l_idxes]
            pre_feature_maps = [p_fm for i, p_fm in enumerate(pre_feature_vars) if i in l_idxes]
            masks = [m for i, m in enumerate(mask_vars) if i in l_idxes]
            rates = [a for i, a in enumerate(rate_vars) if i in l_idxes]

        vul_fmaps = [np.zeros(shape=var.get_shape().as_list()[-1]) for var in feature_maps]

        layer_names = [layer.name[:-2] for layer in layers]
        all_layer_names = [l.name[:-2] for l in all_layer_vars]
        feature_names = [feature.name[:-2] for feature in feature_maps]  # Only the feature maps between all layers
        pre_feature_names = [p_feature.name[:-2] for p_feature in pre_feature_maps]
        mask_names = [mask.name[:-2] for mask in masks]
        rate_names = [rate.name[:-2] for rate in rate_vars]

        self.l_weight_dict = OrderedDict(zip(layer_names, layers))  # {['conv1/weight', variable of conv1/weight], ...}
        self.all_l_weight_dict = OrderedDict(zip(all_layer_names, all_layer_vars))
        self.feature_map_dict = OrderedDict(zip(feature_names, feature_maps))  # {['inference/conv1/f_map', variable of inference/conv1/f_map], ...}
        self.pre_feature_map_dict = OrderedDict(zip(pre_feature_names, pre_feature_maps))
        self.mask_dict = OrderedDict(zip(mask_names, masks))  # {['conv1/p_mask_1', variable of conv1/p_mask_1]}
        self.rate_dict = OrderedDict(zip(rate_names, rates))

        if self.prune_reg == 'channel':
            vul_fmaps = [np.ones(shape=self.params['Dataset']['img_depth'])] + vul_fmaps
            self.vul_fmaps_dict = OrderedDict(zip(feature_names, vul_fmaps[:-1]))  # Input layer as criterion
        elif self.prune_reg == 'filter':
            self.vul_fmaps_dict = OrderedDict(zip(feature_names, vul_fmaps))  # Output layer as criterion
        else:
            self.vul_fmaps_dict = OrderedDict(zip(feature_names, vul_fmaps))
            print('\x1b[4;30m' + 'Warning! Vulnerability based prune does not support Weight/Kernel!' + '\x1b[0m')

        self.rate_names = list(self.rate_dict.keys())

        # Generate rates for saving best rates
        best_rates = []
        for _, rate in enumerate(rates):
            best_rates.append(np.ones(rate.get_shape().as_list()))

        self.best_rate_dict = OrderedDict(zip(rate_names, best_rates))

        # TODO: Make sure if the number of weight layers are same as the number of feature_maps
        assert len(self.l_weight_dict) == len(self.feature_map_dict), 'Weight layers should be equal feature maps'
        assert len(self.l_weight_dict) <= len(self.rate_dict), 'The number of masks should be equal larger than the number of all prunable layers'
        assert len(self.all_l_weight_dict) == len(self.rate_dict), 'The number of all layers is not equal to the number of rates'
        assert len(self.rate_dict) == len(self.mask_dict), 'The number of all masks is not equal to the number of rates'

        self.n_layers = len(self.rate_dict)  # Number of all conv. layers without conv4-3-2: 18

        # Build dictionary to save actions on feature map
        self.min_strategy_dict = {}
        self.strategy_dict = {}

        if self.prune_reg == 'channel' or self.prune_reg == 'filter':
            for i in range(len(self.org_estimates_list)):
                if i == 0:
                    self.min_strategy_dict[i] = [1.0, self.l_bound[0]]

                elif i == self.n_layers-1:              # and self.prune_reg == 'channel'
                    self.min_strategy_dict[i] = [self.l_bound[0], 1.0]

                # elif i == self.n_layers-1 and self.prune_reg == 'filter':
                #     self.min_strategy_dict[i] = [self.l_bound[0], 1.0]

                elif i >= self.n_layers:
                    self.min_strategy_dict[i] = [1.0, 1.0]
                else:
                    if self.rate_names[i] in self.shortcut_names:
                        if self.prune_reg == 'channel':
                            self.min_strategy_dict[i] = [1.0, self.l_bound[0]]
                        elif self.prune_reg == 'filter':
                            self.min_strategy_dict[i] = [self.l_bound[0], 1.0]
                    else:
                        self.min_strategy_dict[i] = [self.l_bound[0], self.l_bound[0]]

                # strategy_dict = [[1, 0.2], [0.2, 0.2], ..., [0.2, 0.2], [0.2, 1], [1, 1]] totally 20 elements for channel pruning
                # strategy_dict = [[1, 0.2], [0.2, 0.2], ..., [0.2, 0.2], [0.2, 1], [1, 1]] totally 20 elements for channel pruning

            self.strategy_dict = self.min_strategy_dict  # dims = 20
            for i in range(len(self.org_estimates_list)):
                self.model_hf.nn[i]['Nif'] = int(self.min_strategy_dict[i][0] * self.original_channels[i][0])
                self.model_hf.nn[i]['Nof'] = int(self.min_strategy_dict[i][1] * self.original_channels[i][1])

        elif self.prune_reg == 'kernel' or self.prune_reg == 'weight':
            for i in range(len(self.org_estimates_list)):
                if i == 0:
                    self.min_strategy_dict[i] = [1.0]
                elif i >= self.n_layers:
                    self.min_strategy_dict[i] = [1.0]
                else:
                    self.min_strategy_dict[i] = [self.l_bound[0]]

            self.strategy_dict = self.min_strategy_dict
            for i in range(len(self.org_estimates_list)):
                self.model_hf.nn[i]['Nif'] = self.original_channels[i][0]
                self.model_hf.nn[i]['Nof'] = self.original_channels[i][1]

    def _build_state_embedding(self):
        layer_embedding = []

        layer_list = list(self.l_weight_dict.values())
        layer_name_list = list(self.l_weight_dict.keys())

        fmap_list = list(self.feature_map_dict.values())

        for layer_ind, w_name in enumerate(list(self.l_weight_dict.keys())):
            this_state = []

            this_state.append(layer_ind)  # layer index (totally 19 layers for ResNet20)

            w_name = layer_name_list[layer_ind]

            kernel_shape = layer_list[layer_ind].get_shape().as_list()  # [k, k, ci, co]

            if layer_ind == 0:
                # Input of CNN
                fmap_shape = [self.params['Meta']['batchsize'], self.params['Dataset']['img_size_x'],
                              self.params['Dataset']['img_size_y'], self.params['Dataset']['img_depth']]
            else:
                fmap_ind = layer_ind - 1  # Index of corresponding input feature map is 1 smaller than layer index
                fmap_shape = fmap_list[fmap_ind].get_shape().as_list()  # [N, h, w, channels] or [N, channels]

            if w_name.find('co') != -1:  # Which means "Conv"

                s = kernel_shape[-1] / kernel_shape[-2] if layer_ind > 0 else 1.0

                this_state.append(kernel_shape[2])  # ci
                this_state.append(kernel_shape[3])  # co
                this_state.append(fmap_shape[1])  # h
                this_state.append(fmap_shape[2])  # w
                this_state.append(s)  # stride
                this_state.append(kernel_shape[1])  # k
                this_state.append(self.org_estimates_list[layer_ind])  # FLOPs

            elif w_name.find('fc') != -1:  # Which means "FC"
                this_state.append(kernel_shape[0])  # ci
                this_state.append(kernel_shape[1])  # co
                if len(fmap_shape) == 4:
                    this_state.append(fmap_shape[1])  # h
                    this_state.append(fmap_shape[2])  # w
                else:
                    this_state.append(1)  # h
                    this_state.append(1)  # w
                this_state.append(0)  # stride
                this_state.append(1)  # k
                this_state.append(self.org_estimates_list[layer_ind])  # FLOPs
            else:
                raise NameError('Variable name is not started from "co" or "fc", current name: %s' % w_name)  # Please check the

            # this 3 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}

            layer_embedding.append(np.array(this_state))

        layer_embedding = np.array(layer_embedding)
        assert len(layer_embedding.shape) == 2, 'layer_embedding shape is in error'

        # State normalization
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding  # [conv2_1_1, con2_1_2, ..., conv4_3_3, fc6]

    def _get_vulnerability(self, vul_fmaps_dict):

        fmap_names = list(self.vul_fmaps_dict.keys())

        if self.prune_reg == 'channel':
            start_ind = 1
        else:
            start_ind = 0

        for i in range(start_ind, len(fmap_names)):
            fmap_name = fmap_names[i]
            if self.prune_reg == 'channel':
                fmap_name_pre = fmap_names[i-1]
                self.vul_fmaps_dict[fmap_name] = vul_fmaps_dict[fmap_name_pre]
            else:
                self.vul_fmaps_dict[fmap_name] = vul_fmaps_dict[fmap_name]

    def get_target_rate(self):
        target_rate = self.params['Prune_config']['target_sparsity']
        if self.sparsity_mode == 'flops' and self.prune_reg in ['channel', 'filter']:
            W_rest = 0.0
            W_all = 0.0
            for i, l_flops in enumerate(self.org_estimates_list):
                W_all += l_flops
                if i == 0:
                    # Only output channels will be pruned (1st layer)
                    W_rest += l_flops * target_rate
                elif i == len(self.org_estimates_list)-1:
                    # Only input channels will be pruned (last layer)
                    W_rest += l_flops * target_rate
                else:
                    W_rest += l_flops * target_rate * target_rate

            target_params_rate = W_rest/W_all
        else:
            target_params_rate = target_rate

        return target_params_rate


    def _target_action(self):
        W_all = self.org_estimates
        W_rest = 0
        W_reduced = 0
        W_t = 0

        # for kernel_prune and weight_prune, we can use a ration to estimate the  hw_flow
        # right now the cur_estimates_list is original estimate list

        for i in range(len(self.org_estimates_list)):
            flops = self.org_estimates_list[i]
            buffer_flops = self.buffer_list[i]

            li_name = self.model_hf.nn[i]['name']

            l_id = self.backbone_ids[self.cur_ind]
            l_pre_id = self.backbone_ids[self.cur_ind-1]
            l_post_id = self.backbone_ids[self.cur_ind+1] if self.prune_reg == 'filter' else None

            if self.prune_reg == 'channel' or self.prune_reg == 'filter':
                if i == l_id:
                    if self.prune_reg == 'channel':
                        # Current conv layer is pruned filter-wise with max sparsity ratio
                        self.model_hf.nn[i]['Nif'] = self.original_channels[i][0]
                        W_t += flops * self.strategy_dict[i][1]
                        W_t += buffer_flops

                    elif self.prune_reg == 'filter':
                        # Current conv layer is pruned channel-wise with max sparsity ratio
                        self.model_hf.nn[i]['Nof'] = self.original_channels[i][1]
                        W_t += flops * self.strategy_dict[i][1]
                        W_t += buffer_flops
                elif i == l_pre_id and self.prune_reg == 'channel':
                    # Previous conv layer is pruned_channel-wise with max sparsity ratio
                    self.model_hf.nn[i]['Nof'] = self.original_channels[i][1]
                    W_t += flops * self.strategy_dict[i][0]
                    W_reduced += buffer_flops * self.strategy_dict[i][0]

                elif li_name == self.model_hf.nn[l_id]['ch_shortcut'] and self.prune_reg == 'channel':
                    self.model_hf.nn[i]['Nif'] = self.original_channels[i][0]
                    W_t += flops * self.strategy_dict[i][1]
                    W_reduced += buffer_flops * self.strategy_dict[l_id][1]

                elif li_name == self.model_hf.nn[l_id]['pre_shortcut'] and self.prune_reg == 'channel':
                    self.model_hf.nn[i]['Nof'] = self.original_channels[i][1]
                    W_t += flops * self.strategy_dict[i][0]
                    W_reduced += buffer_flops * self.strategy_dict[i][0]

                # for fitler pruning, consider next layer
                elif i == l_post_id and self.prune_reg == 'filter':
                    # next conv layer is prune filter-wise with max sparsity ratio
                    self.model_hf.nn[i]['Nif'] = self.original_channels[i][0]
                    W_t += flops * self.strategy_dict[i][1]
                    W_reduced += buffer_flops * self.strategy_dict[i][1]

                elif li_name == self.model_hf.nn[l_id]['fi_shortcut'] and self.prune_reg == 'filter':
                    self.model_hf.nn[i]['Nof'] = self.original_channels[i][1]
                    W_t += flops * self.strategy_dict[i][0]
                    W_reduced += buffer_flops * self.strategy_dict[i][0]

                elif li_name == self.model_hf.nn[l_id]['post_shortcut'] and self.prune_reg == 'filter':
                    self.model_hf.nn[i]['Nif'] = self.original_channels[i][0]
                    W_t += flops * self.strategy_dict[l_id][1]
                    W_reduced += buffer_flops * self.strategy_dict[l_id][1]

                elif i <= l_pre_id:
                    W_reduced += flops * self.strategy_dict[i][0] * self.strategy_dict[i][1]
                    W_reduced += buffer_flops * self.strategy_dict[i][0]

                else:
                    W_rest += flops * self.strategy_dict[i][0] * self.strategy_dict[i][1]
                    W_rest += buffer_flops * self.strategy_dict[i][0]

            elif self.prune_reg == 'kernel' or self.prune_reg == 'weight':

                if i == l_id:
                    W_t += flops
                    W_t += buffer_flops

                elif i <= l_pre_id:
                    W_reduced += flops * self.strategy_dict[i][0]
                else:
                    W_rest += flops * self.strategy_dict[i][0]
                    W_rest += buffer_flops

        W_duty = self.target_rate * W_all - (W_rest * 1.0 + W_reduced)
        target_action = W_duty / W_t * 1.0

        return target_action

    def _target_sparse_act(self):
        W_all = 0
        W_rest = 0
        W_reduced = 0
        W_t = 0

        for i in range(len(self.org_estimates_list)):

            l_k = self.model_hf.nn[i]['k']
            if l_k is None:
                l_k = 1  # In case layer i is FC layer
            l_cin = self.original_channels[i][0]
            l_cout = self.original_channels[i][1]

            # Count sparse parameters according to different sparsity mode
            if self.sparsity_mode == 'structure':
                if self.prune_reg == 'channel':
                    l_params = l_cin
                elif self.prune_reg == 'filter':
                    l_params = l_cout
                elif self.prune_reg == 'kernel':
                    l_params = l_cin * l_cout
                else:
                    l_params = l_k * l_k * l_cin * l_cout
            elif self.sparsity_mode == 'parameter':
                l_params = l_k * l_k * l_cin * l_cout
            else:
                raise NameError('Sparsity mode "{}" does not exist in ["structure", "parameter", "flops"] !'.format(self.sparsity_mode))

            W_all += l_params

            li_name = self.model_hf.nn[i]['name']

            l_id = self.backbone_ids[self.cur_ind]

            if self.prune_reg == 'channel' or self.prune_reg == 'filter':

                if i == l_id:
                    W_t += l_params

                elif li_name == self.model_hf.nn[l_id]['ch_shortcut'] or li_name == self.model_hf.nn[l_id]['fi_shortcut']:
                    W_t += l_params

                elif i < l_id:
                    if self.prune_reg == 'channel':
                        W_reduced += l_params * self.strategy_dict[i][0]
                    elif self.prune_reg == 'filter':
                        W_reduced += l_params * self.strategy_dict[i][1]

                else:
                    if self.prune_reg == 'channel':
                        W_rest += l_params * self.strategy_dict[i][0]
                    elif self.prune_reg == 'filter':
                        W_rest += l_params * self.strategy_dict[i][1]

            elif self.prune_reg == 'kernel' or self.prune_reg == 'weight':

                if i == l_id:
                    W_t += l_params

                elif i < l_id:
                    W_reduced += l_params * self.strategy_dict[i][0]
                else:
                    W_rest += l_params * self.strategy_dict[i][0]

        W_duty = self.target_rate * W_all - (W_rest * 1.0 + W_reduced)
        target_action = W_duty / W_t * 1.0

        return target_action

    def _action_wall(self):
        def action_wall_op():
            action_soll = tf.clip_by_value(self.target_action, clip_value_min=self.l_bound[0], clip_value_max=self.l_bound[1])
            action_wall_op = tf.minimum(self.action, action_soll)  # Because here action means preserve ratio

            return action_wall_op

        self.action_op = action_wall_op()

    def _if_last_layer(self):
        return len(self.l_weight_dict) == self.cur_ind + 1

    def _progress_bar(self):
        layer = self.cur_ind
        num_layers = len(self.l_weight_dict)-1
        bar_len = num_layers
        bars = int((float(layer) / num_layers) * bar_len)
        sys.stdout.write('\r (%d/%d)' % (layer, num_layers))
        for b in range(bars):
            sys.stdout.write('|')
        sys.stdout.flush()

    def _reset_all_masks(self):
        # Reset all masks
        mask_var_list = list(self.mask_dict.values())  # Get all mask variables
        all_layer_list = list(self.all_l_weight_dict.values())

        mask_reset_ops = []
        for i, m in enumerate(mask_var_list):
            m_name = '/'.join(m.name.split('/')[:-1])
            w_var = all_layer_list[i]
            w_var_name = '/'.join(w_var.name.split('/')[:-1])
            assert m_name == w_var_name, 'Mask reset requires the correspondence between mask and weight'

            m_shape = m.get_shape().as_list()
            if w_var_name.find('fc') != -1:
                m_cut_dims = tf.squeeze(tf.where(tf.equal(m_shape, [1, 1])))
            else:
                m_cut_dims = tf.squeeze(tf.where(tf.equal(m_shape, [1, 1, 1, 1])))

            m_init = tf.reduce_sum(tf.abs(w_var), axis=m_cut_dims)
            m_init = m_init / tf.reduce_max(m_init)
            m_init = tf.reshape(m_init, m.shape)

            mask_reset_ops.extend([m.assign(m_init)])

        self.mask_init_op = mask_reset_ops

        for layers in range(self.n_layers):
            self.model_hf.nn[layers]['Nif'] = self.original_channels[layers][0]
            self.model_hf.nn[layers]['Nof'] = self.original_channels[layers][1]

    def _reset_all_rates(self):
        # Reset all prune rates
        rate_var_list = list(self.rate_dict.values())  # Get all mask variables

        self.rate_reset_op = [tf.assign(rate_var, 1.0) for rate_var in rate_var_list]

    def _rescale_cur_action(self, layer_action):
        [lbound, rbound] = self.params['DDPG']['actor_bounds']
        action_out = layer_action * (rbound - lbound) + lbound
        return action_out

    def _calc_cur_action(self, layer_action, p_mask_var):
        mask_shape = p_mask_var.get_shape().as_list()
        mask_length = float(np.prod(mask_shape))

        preserved_ch = np.round(mask_length * layer_action)
        preserved_ch = np.ceil(preserved_ch * 1.0)
        preserved_ch = preserved_ch.squeeze()
        preserved_ch = np.maximum(preserved_ch, 1.0)  # In case that layer is totally pruned
        cur_action = np.float32(preserved_ch/mask_length)

        return cur_action

    def _global_pruning(self):

        layer_list = list(self.l_weight_dict.values())

        rate_var_list = list(self.rate_dict.values())  # Get all layer compression ratio
        rate_var_names = list(self.rate_dict.keys())

        updated_rates = OrderedDict()

        for i, l in enumerate(layer_list):

            l_id = self.backbone_ids[i]

            _rate_name = rate_var_names[l_id]
            _rate = rate_var_list[l_id]
            _cur_act = self.action_list[i]

            updated_rates[_rate_name] = _cur_act

            if self.prune_reg == 'channel':
                if self.model_hf.nn[l_id]['ch_shortcut'] is not None:
                    shortcut_name = '/'.join(self.model_hf.nn[l_id]['ch_shortcut'].split('/')[:-1])
                    for r in rate_var_list:
                        if r.name.find(shortcut_name) != -1:
                            cur_rate_name = r.name.split(':')[0]
                            updated_rates[cur_rate_name] = _cur_act
                            break

            elif self.prune_reg == 'filter':
                if self.model_hf.nn[l_id]['fi_shortcut'] is not None:
                    shortcut_name = '/'.join(self.model_hf.nn[l_id]['fi_shortcut'].split('/')[:-1])
                    for r in rate_var_list:
                        if r.name.find(shortcut_name) != -1:
                            cur_rate_name = r.name.split(':')[0]
                            updated_rates[cur_rate_name] = _cur_act
                            break

        self.global_prune_op = []
        for rate_name in rate_var_names:
            self.global_prune_op.extend([self.rate_dict[rate_name].assign(updated_rates[rate_name])])

    def _apply_best_prune(self):
        rate_var_names = list(self.rate_dict.keys())

        best_rates_op = []

        for rate_name in rate_var_names:
            best_rates_op.extend([self.rate_dict[rate_name].assign(self.best_rate_dict[rate_name])])

        if self.prune_reg == 'channel' or self.prune_reg == 'filter':
            for i in range(len(self.org_estimates_list)):
                self.model_hf.nn[i]['Nif'] = int(self.best_strategy_dict[i][0] * self.original_channels[i][0])
                self.model_hf.nn[i]['Nof'] = int(self.best_strategy_dict[i][1] * self.original_channels[i][1])

        return best_rates_op

    def update_best_rates(self, best_rates):
        for i, name in enumerate(list(self.best_rate_dict.keys())):
            self.best_rate_dict[name] = best_rates[i]

    def reset(self):
        self.cur_ind = self.init_ind
        self.strategy = [1.0] if self.prune_reg == 'channel' and (not self.prune_1st) else []  # To save actions in one round
        self.actions = np.ones(len(self.l_weight_dict), dtype=np.float32)
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)  # To save input and output channels
        self.model_hf.nn = copy.deepcopy(self.model_nn_org)

        if self.prune_reg == 'channel' or self.prune_reg == 'filter':
            for i in range(len(self.org_estimates_list)):
                self.model_hf.nn[i]['Nif'] = int(self.min_strategy_dict[i][0] * self.original_channels[i][0])
                self.model_hf.nn[i]['Nof'] = int(self.min_strategy_dict[i][1] * self.original_channels[i][1])

        self.layer_embedding[:, -1] = 1.0  # reduced = 1: FLOPs fully reduced
        self.layer_embedding[:, -2] = 0.0  # rest = 0.0
        self.layer_embedding[:, -3] = 0.0  # a_{t-1} = 0.0

        obs = self.layer_embedding[0]
        obs[-2] = sum(self.org_estimates_list[1:]) * 1.0 / self.org_estimates

        return obs


    def _validate(self, sess, is_training):

        acc1, acc5, _, = evaluate(sess, self.model, self.attack, self.aa_ops, self.params, self.ds_switch, self.dataset,
                                 is_training, bn_mode=BN_TRAIN_MODE, print_acc=False)  # , run_aa_eval=True)

        bng_acc1, bng_acc5, _ = evaluate(sess, self.model, self.attack, self.aa_ops, self.params, self.ds_switch,
                                         self.dataset, is_training, bn_mode=BN_TRAIN_MODE, print_acc=False,
                                         run_benign=True)

        if self.params['Prune_config']['acc'] == 'acc1':
            print('Top1  bng_acc = {:6.2f}%  adv_acc = {:6.2f}%'.format(bng_acc1, acc1))
            return acc1 * 0.01, bng_acc1*0.01
        elif self.params['Prune_config']['acc'] == 'acc5':
            print('Top5  bng_acc = {:6.2f}%  adv_acc = {:6.2f}%'.format(bng_acc5, acc5))
            return acc5 * 0.01, bng_acc5*0.01
        else:
            raise NotImplementedError('Please recheck "acc" in "Prune_config" !')

    def step(self, sess, action, episode, is_training, save_var_list):
        #initial a current_estimates for kernel and weight pruning
        current_estimates = 0
        mask_var_list = list(self.mask_dict.values())  # Get all mask variables

        l_id = self.backbone_ids[self.cur_ind]
        l_pre_id = self.backbone_ids[self.cur_ind - 1] if self.prune_reg == 'channel' else None
        l_post_id = self.backbone_ids[self.cur_ind + 1] if self.prune_reg == 'filter' else None

        # Print pruning progress
        if self.cur_ind == self.init_ind:
            print('Pruning Progress:')
        self._progress_bar()

        # Compute action wall
        action = self._rescale_cur_action(action)  # Rescale action into actor_bounds range

        if self.params['Prune_config']['reward_mode'] in ['acc', 'acc_aa', 'acc+aa']:
            # Prepare for action_wall
            if self.sparsity_mode == 'flops':
                tar_action = self._target_action()
            else:
                tar_action = self._target_sparse_act()
            action_out = sess.run(self.action_op, feed_dict={self.action: action, self.target_action: tar_action})
        else:
            action_out = action

        cur_p_mask = mask_var_list[l_id]
        cur_action = self._calc_cur_action(action_out, cur_p_mask)

        self.actions[self.cur_ind] = cur_action

        # Attention!!! Length of strategy_dict is equal to the number of all layers (conv1 to fc6)

        if self.prune_reg == 'channel':
            self.strategy_dict[l_pre_id][1] = cur_action  # compression ratio on previous layer e.g. for conv1/f_map, here is conv1/weight
            self.strategy_dict[l_id][0] = cur_action  # compression ratio on next layer e.g. for conv1/f_map, here is conv2_1/weight
            self.strategy.append(cur_action)
            #update hwflow args based on the strategy dict
            self.model_hf.nn[l_pre_id]['Nof'] = int(np.round(cur_action * self.original_channels[l_pre_id][1]))
            self.model_hf.nn[l_id]['Nif'] = int(np.round(cur_action * self.original_channels[l_id][0]))

            if self.model_hf.nn[l_id]['ch_shortcut'] is not None:
                shortcut_name = self.model_hf.nn[l_id]['ch_shortcut']
                for id, l in enumerate(self.model_hf.nn):
                    if l['name'] == shortcut_name:
                        self.model_hf.nn[id]['Nif'] = int(np.round(cur_action * self.original_channels[id][0]))
                        break

            if self.model_hf.nn[l_id]['pre_shortcut'] is not None:
                shortcut_name = self.model_hf.nn[l_id]['pre_shortcut']
                for id, l in enumerate(self.model_hf.nn):
                    if l['name'] == shortcut_name:
                        self.model_hf.nn[id]['Nof'] = int(np.round(cur_action * self.original_channels[id][1]))
                        break

        elif self.prune_reg == 'filter':
            self.strategy_dict[l_id][1] = cur_action  # compression ratio on this layer e.g. for conv1/f_map, here is conv1/weight
            self.strategy_dict[l_post_id][0] = cur_action  # compression ration on next layer e.g. for conv1/f_map, here is conv2_1/weight
            self.strategy.append(cur_action)
            #update hwflow args based on the strategy dict
            self.model_hf.nn[l_id]['Nof'] = int(np.round(cur_action * self.original_channels[l_id][1]))
            self.model_hf.nn[l_post_id]['Nif'] = int(np.round(cur_action * self.original_channels[l_post_id][0]))

            if self.model_hf.nn[l_id]['fi_shortcut'] is not None:
                shortcut_name = self.model_hf.nn[l_id]['fi_shortcut']
                for id, l in enumerate(self.model_hf.nn):
                    if l['name'] == shortcut_name:
                        self.model_hf.nn[id]['Nof'] = int(np.round(cur_action * self.original_channels[id][1]))
                        break

            if self.model_hf.nn[l_id]['post_shortcut'] is not None:
                shortcut_name = self.model_hf.nn[l_id]['post_shortcut']
                for id, l in enumerate(self.model_hf.nn):
                    if l['name'] == shortcut_name:
                        self.model_hf.nn[id]['Nif'] = int(np.round(cur_action * self.original_channels[id][0]))
                        break

        elif self.prune_reg == 'kernel' or self.prune_reg == 'weight':
            self.strategy_dict[l_id][0] = cur_action
            self.strategy.append(cur_action)

        if self._if_last_layer():
            assert len(self.strategy) == self.cur_ind+1

            # Implement global pruning
            all_rates = sess.run(self.global_prune_op, feed_dict={self.action_list: self.actions})

            # if kernel or weight pruning, use stategy to caculate FLOPs
            if self.prune_reg == 'kernel' or self.prune_reg == 'weight':
                for i in range(len(self.org_estimates_list)):
                    current_estimates_tmp = self.org_estimates_list[i] * self.strategy_dict[i][0]
                    current_estimates += current_estimates_tmp

            else:
                current_estimates = self._cur_estimates()

            preserve_ratio = current_estimates * 1.0 / self.org_estimates

            cur_aa_acc, cur_bng_acc = self._validate(sess, is_training)

            reward = acc_estimates_reward(cur_bng_acc, self.params, acc_aa=cur_aa_acc)

            if reward > self.params['DDPG']['min_reward'] or episode > self.params['DDPG']['warmup']:
                print('Estimate Reduction Factor: %.3f' % preserve_ratio)
                print('All layer compression:', np.squeeze(self.strategy))

            ##Calculate the estimates one final time

            try:
                log = open(os.path.join(self.params['Meta']['output_folder'], 'Episode_wise_Rewards.csv'), 'a')
            except:
                log = open(os.path.join(self.params['Meta']['output_folder'], 'Episode_wise_Rewards.csv'), 'w')
                log.write('.......................\n')
                log.write('Iter\tIter_reward\tEstimate_Reduction\tBng_Acc\tAdv_Acc\torg_ops\tnew_ops\titer_strategy\n')

            log.write(('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' % (
                episode, reward, preserve_ratio, cur_bng_acc, cur_aa_acc, self.original_ops, current_estimates)) + str(self.strategy) + '\n')

            # If see better reward
            if reward > self.best_reward and episode > self.params['DDPG']['warmup']:
                self.best_reward = reward
                self.best_acc = cur_bng_acc
                self.best_estimate = current_estimates
                self.best_strategy = self.strategy.copy()
                self.best_strategy_dict = self.strategy_dict
                self.update_best_rates(all_rates)

                # Save pure pruned model
                saver = tf.train.Saver(save_var_list)
                checkpoint_file = os.path.join(self.params['Meta']['output_folder'],
                                               self.params['Meta']['output_name'] + '_best_robust_pruned.ckpt')
                saver.save(sess, checkpoint_file)

            ##Calculate the estimates one final time

            try:
                log = open(os.path.join(self.params['Meta']['output_folder'], 'Best_Rewards.csv'), 'a')
            except:
                log = open(os.path.join(self.params['Meta']['output_folder'], 'Best_Rewards.csv'), 'w')
                log.write('.......................\n')
                log.write('Iter\tBest_reward\tEstimate_Reduction\tBng_Acc\tAdv_Acc\torg_ops\tnew_ops\tbest_strategy\n')

            log.write(('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' % (
                episode, self.best_reward, preserve_ratio  ,cur_bng_acc, cur_aa_acc, self.original_ops, current_estimates)) + str(self.best_strategy) + '\n')

            log.close()

            obs = self.layer_embedding[self.cur_ind, :].copy()
            done = True

            return obs, reward, done

        reward = 0
        done = False

        self.cur_ind += 1  # Move to next layer

        # Build next state vector
        self.layer_embedding[self.cur_ind][-3] = self._cur_estimates_reduced()*1.0 / self.org_estimates
        self.layer_embedding[self.cur_ind][-2] = sum(self.org_estimates_list[self.cur_ind+1:] + self.buffer_list[self.cur_ind+1:]) * 1.0 / self.org_estimates
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]

        obs = self.layer_embedding[self.cur_ind, :].copy()  # Give the next layer observation as output

        return obs, reward, done
