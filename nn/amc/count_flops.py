import tensorflow as tf
import numpy as np
from collections import OrderedDict


'''
    @article{FLOPs_estimation,
      author    = {Pavlo Molchanov and
                   Stephen Tyree and
                   Tero Karras and
                   Timo Aila and
                   Jan Kautz},
      title     = {Pruning Convolutional Neural Networks for Resource Efficient Transfer
                   Learning},
      journal   = {CoRR},
      volume    = {abs/1611.06440},
      year      = {2016},
      url       = {http://arxiv.org/abs/1611.06440},
      archivePrefix = {arXiv},
      eprint    = {1611.06440},
      timestamp = {Mon, 13 Aug 2018 16:46:56 +0200},
      biburl    = {https://dblp.org/rec/journals/corr/MolchanovTKAK16.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
'''


def _conv_flops(layer_shape, f_map_shape, count_fmap=False):
    [k, k, ci, co] = layer_shape  # [k*k*ci*co]
    [h, w, _] = f_map_shape
    if count_fmap:
        l_flops = 2 * h * w * (ci * k * k + 1) * co
    else:
        l_flops = k * k * ci * co

    return l_flops


def _fc_flops(layer_shape, count_fmap=False):
    [ci, co] = layer_shape
    if count_fmap:
        l_flops = (2 * ci - 1) * co
    else:
        l_flops = ci*co

    return l_flops
    

def _relu_flops(relu_shape, count_fmap=False):
    if len(relu_shape) > 2:
        [h, w, co] = relu_shape
        relu_flops = h * w * co if count_fmap else co
    else:
        relu_flops = relu_shape[-1]

    return relu_flops


def _pool_flops(pool_shape, pre_fmp_map, count_fmap=False):
    [h_o, w_o, co] = pool_shape
    [h_i, w_i, _] = pre_fmp_map
    k_p = int(h_i / h_o)
    pool_flops = h_i * w_i * k_p * k_p * co if count_fmap else co

    return pool_flops


def get_flops(layer_list, count_buffer=True):

    layer_flops = []
    buffer_flops = []
    layer_io_channels = []

    for layer in layer_list:
        l_name = layer['name']
        ci, co = layer['Nif'], layer['Nof']
        k = layer['k']
        l_shape = [ci,co] if l_name.find('fc') != -1 else [k,k,ci,co]
        layer_io_channels.append(l_shape[-2:])  # Put [channels, filters] into "layer_io_channels"

        buf_flops = 0.0
        if l_name.find('conv') != -1:
            fp_shape = layer['f_map']
            l_flops = _conv_flops(l_shape, fp_shape)

            act = layer['act']
            act_map = layer['act_map']
            pool = layer['pool']
            pool_map = layer['pool_map']
            pre_f_map = layer['pre_f_map']

            if act:
                buf_flops += _relu_flops(act_map) if act == 'Relu' else 0

            if pool:
                buf_flops += _pool_flops(pool_map, pre_f_map)

        else:
            l_flops = _fc_flops(l_shape)

        layer_flops.append(l_flops)
        buffer_flops.append(buf_flops)

    if count_buffer:
        total_flops = sum(layer_flops) + sum(buffer_flops)
    else:
        total_flops = sum(layer_flops)
        buffer_flops = [0.0] * len(buffer_flops)

    return total_flops, layer_flops, buffer_flops, layer_io_channels




