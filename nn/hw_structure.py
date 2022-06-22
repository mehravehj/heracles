import os
import tensorflow as tf

from nn.amc.count_flops import get_flops

class HW_Structure(object):
    def __init__(self, layer_list, f_map_list, act_list):
        self.layer_list = layer_list
        self.f_map_list = f_map_list
        self.act_list = act_list
        self.nn = self._model_struct()

    def _model_struct(self):
        nn_struct = []

        last_shortcut = None

        block_name = '/'.join(self.layer_list[0].name.split('/')[:2])

        for i, layer in enumerate(self.layer_list):

            l_shape = layer.get_shape().as_list()
            l_f_map = self.f_map_list[i].get_shape().as_list()[1:]

            # Update block name, Add block shortcut layer to fi_shortcut of last block layer
            if layer.name.find(block_name) == -1:
                nn_struct[-1]['fi_shortcut'] = last_shortcut
                last_shortcut = None

                block_name = '/'.join(layer.name.split('/')[:2])

            if self.f_map_list[i-1].name.find('shortcut') != -1:
                last_shortcut = self.layer_list[i-1].name
                l_pre_fmap = self.f_map_list[i-2].get_shape().as_list()[1:]
                l_pre_name = self.layer_list[i-2].name
                l_ch_short = self.layer_list[i-1].name
            else:
                l_pre_fmap = self.f_map_list[i-1].get_shape().as_list()[1:] if i != 0 else None
                l_pre_name = self.layer_list[i-1].name
                l_ch_short = None

            try:
                nn_struct[-2]['post_shortcut'] = l_ch_short
            except:
                pass

            l_pre_short = nn_struct[-1]['fi_shortcut'] if len(nn_struct) != 0 else None

            l_name = layer.name

            l_is_fc = True if l_name.find('fc') != -1 else False
            l_id = l_name.split('/')[-2]
            l_act = None
            l_act_shape = None
            l_pool = None
            l_pool_shape = None

            for act in self.act_list:
                if act.name.find(l_id) != -1:
                    if act.name.find('Relu') != -1:
                        l_act = 'Relu'
                        l_act_shape = act.get_shape().as_list()[1:]
                    elif act.name.find('pool') != -1:
                        l_pool = 'avg_pool' if act.name.find('avg_pool') != -1 else 'max_pool'
                        l_pool_shape = act.get_shape().as_list()[1:]

            l_struct = {'Nif': l_shape[-2],
                        'Nof': l_shape[-1],
                        'k': None if l_is_fc else l_shape[0],
                        'f_map': l_f_map,
                        'pre_f_map': l_pre_fmap,
                        'name': l_name,
                        'pre_name': l_pre_name,
                        'pre_shortcut': l_pre_short,
                        'post_shortcut': None,
                        'ch_shortcut': l_ch_short,
                        'fi_shortcut':  None,
                        'act': l_act,
                        'act_map': l_act_shape,
                        'pool': l_pool,
                        'pool_map': l_pool_shape}
            nn_struct.append(l_struct)

        return nn_struct

    def _update_struct(self):
        for i, layer in enumerate(self.nn):
            if layer['pre_f_map']:
                layer['pre_f_map'][-1] = layer['Nif']

            layer['f_map'][-1] = layer['Nof']

            if layer['act_map']:
                layer['act_map'][-1] = layer['Nof']

            if layer['pool_map']:
                layer['pool_map'][-1] = layer['Nof']

            self.nn[i] = layer

    def start_estimate(self, count_buffer=True):
        self._update_struct()  # Update all layer with current compression
        total_flops, layer_flops, buffer_flops, _ = get_flops(self.nn, count_buffer)
        start_estimate = {'total_ops': total_flops, 'layer_ops': layer_flops}

        return start_estimate
