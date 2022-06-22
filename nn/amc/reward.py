import numpy as np


def acc_estimates_reward(acc, config, acc_aa=0.0):
    reward_mode = config['Prune_config']['reward_mode']
    reward_mode_list = ['acc', 'acc_aa', 'acc+aa']
    assert reward_mode in reward_mode_list, f'{reward_mode} not exist! Please check reward mode in config file !'

    aa = acc_aa

    if reward_mode == 'acc':
        return acc

    elif reward_mode == 'acc_aa':
        return acc * np.exp(aa)

    elif reward_mode == 'acc+aa':
        aa_beta = config['Prune_config']['aa_beta']
        return acc + aa_beta * aa
