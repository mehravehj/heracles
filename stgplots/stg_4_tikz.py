import json
import numpy as np
import copy
import os
import logging
import argparse

parser = argparse.ArgumentParser(description='Heracles strategies to tikz visualization')
parser.add_argument('--arch', type=str, default='vgg16_bn', choices=("vgg16_bn", "resnet18", "wrn_28_4", "ResNet50"), help='DNN architecture')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=("CIFAR10", "SVHN", "imagenet"), help='dataset name')
parser.add_argument('--prune_reg', type=str, default='weight', choices=('weight', 'channel'), help='pruning regularity')
parser.add_argument('--p_rate', type=float, default=0.1, help='target pruning rate that is used to name strategies')

args = parser.parse_args()


def _to_errorline(args, dec_pos=3):
    ds_name = args.dataset
    cwd_dir = os.getcwd()

    strategy_f = '/'.join((cwd_dir.split('/')[:-1]+['/strategies/{}.json'.format(ds_name)]))
    strategies = json.load(open(strategy_f))
    p_rate = str(args.p_rate).replace('.', '')
    if args.p_rate == 0.1 and args.prune_reg == 'weight':
        p_rate += '0'
    print(f'\n>>>>> {args.prune_reg} pruning on {args.arch} {args.dataset} with rate {args.p_rate}')

    log_dir = f'./{args.dataset}'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_fname = f"{args.arch}_{args.prune_reg}_{p_rate}.csv"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(log_dir, log_fname), "w")
    )

    stg_list = []

    stgs = strategies[args.arch][args.prune_reg]
    for stg_id in stgs.keys():
        if stg_id.find('uni') == -1 and stg_id.find(p_rate) != -1:
            stg_list.append(stgs[stg_id])
            print(f'add strategy: {stg_id}')

    stg_list = np.swapaxes(stg_list, 0, 1)
    p_stgs = copy.copy(stg_list)

    bias = 0
    print('\n>>>>> Generating code for tikz plot\n')
    logger.info(f"layer\tmean\tmin\tmax")
    for i in range(p_stgs.shape[0]):
        layer_rates = p_stgs[i]
        median_rate = np.median(layer_rates)
        min_err = median_rate - np.min(layer_rates)
        max_err = np.max(layer_rates) - median_rate

        if args.arch == 'resnet18' and args.prune_reg == 'channel':
            if i in [5, 9, 13]:
                logger.info(
                    f"{i + bias + 1}\t{np.round(median_rate, dec_pos)}\t{np.round(min_err, dec_pos)}\t{np.round(max_err, dec_pos)}")
                bias += 1

        if args.arch == 'wrn_28_4' and args.prune_reg == 'channel':
            if i in [1, 9, 17]:
                logger.info(
                    f"{i + bias + 1}\t{np.round(median_rate, dec_pos)}\t{np.round(min_err, dec_pos)}\t{np.round(max_err, dec_pos)}")
                bias += 1

        if args.arch == 'ResNet50' and args.prune_reg == 'channel':
            if i in [1, 10, 22, 40]:
                logger.info(
                    f"{i + bias + 1}\t{np.round(median_rate, dec_pos)}\t{np.round(min_err, dec_pos)}\t{np.round(max_err, dec_pos)}")
                bias += 1

        logger.info(f"{i + bias + 1}\t{np.round(median_rate, dec_pos)}\t{np.round(min_err, dec_pos)}\t{np.round(max_err, dec_pos)}")


if __name__ == '__main__':
    _to_errorline(args)
