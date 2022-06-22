import tensorflow as tf
import sys
import uuid
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict

BN_PERCEPTION_MODE = 0
BN_TRAIN_MODE = 1


def _dataset_size(ds, dataset_switch, params):

    if dataset_switch == ds.TRAIN_SET:
        dataset_size = params['Dataset']['num_train_samples']
        # print("\n\x1b[0;30;45m" + "Train Set Evaluation" + "\x1b[0m")
    elif dataset_switch == ds.VALID_SET:
        dataset_size = params['Dataset']['num_valid_samples']
        # print("\n\x1b[0;30;44m" + "Valid Set Evaluation" + "\x1b[0m")
    elif dataset_switch == ds.TEST_SET:
        dataset_size = params['Dataset']['num_test_samples']
        # print("\n\x1b[0;30;42m" + "Test Set Evaluation" + "\x1b[0m")
    else:
        raise NameError('Please check dataset switch name !')

    return dataset_size


def _progress_bar(step, num_steps):
    step = step + 1
    bar_len = round(num_steps/4)
    bars = int((float(step) / num_steps) * bar_len)
    sys.stdout.write('\r(%d/%d) ' % (step, num_steps))
    for b in range(bars):
        sys.stdout.write('▋')
    sys.stdout.flush()


def eval_classification(sess, eval_op, params, ds_switch, ds, is_training, bn_mode=BN_PERCEPTION_MODE, print_acc=False):

    dataset_size = _dataset_size(ds, ds_switch, params)

    if dataset_size == 0:
        raise NameError('Dataset size can not be 0. Please check the configuration file again!')

    steps_per_epoch = dataset_size // params['Meta']['batchsize']
    acc1_all = []
    acc5_all = []
    losses = []
    reg_losses = []
    for step in range(steps_per_epoch):
        acc_1, acc_5, preds, labs, step_loss, step_regloss = sess.run(eval_op, feed_dict={is_training: bn_mode, ds.get_switch(): ds_switch})
        acc1_all.append(acc_1)
        acc5_all.append(acc_5)
        losses.append(step_loss)
        reg_losses.append(step_regloss)

    acc1_avg = np.mean(acc1_all)
    acc5_avg = np.mean(acc5_all)
    loss_avg = np.mean(losses)
    regloss_avg = np.mean(reg_losses)

    if print_acc:
        print('\x1b[0;30m' + '[Top1] =' + '\x1b[0m', '\x1b[4;30m' + '%.2f' % acc1_avg + ' %' + '\x1b[0m', end='\n')

    return acc1_avg, acc5_avg, loss_avg, regloss_avg


def eval_classification_attack(sess, model, attack, aa_ops, params, ds_switch, ds, is_training, bn_mode=BN_PERCEPTION_MODE,
                               print_acc=True, run_benign=False, run_aa_eval=False, verbose=False):

    dataset_size = _dataset_size(ds, ds_switch, params)

    if dataset_size != 0:
        steps_per_epoch = dataset_size // params['Meta']['batchsize']

        # Define attack operations
        attack_reset = aa_ops['reset_attack']
        attack_get_inputs = aa_ops['get_inputs']

        aa_init_feed = {ds.get_switch(): ds_switch, attack.aa_on: False}
        aa_on_feed = {is_training: bn_mode, attack.aa_on: True, attack.aa_init_on: False}

        attack_steps = params['Attack_config']['steps']
        attack_op = aa_ops['get_attack']

        acc1_all = []
        acc5_all = []

        losses = []

        if verbose:
            if ds_switch == ds.TRAIN_SET:
                if run_benign:
                    print('\n>> Evaluating benign training set')
                else:
                    print('\n>> Evaluating attack training set')

            if ds_switch == ds.VALID_SET:
                if run_benign:
                    print('\n>> Evaluating benign valid set')
                else:
                    print('\n>> Evaluating attack valid set')

            if ds_switch == ds.TEST_SET:
                if run_benign:
                    print('\n>> Evaluating benign test set')
                else:
                    print('\n>> Evaluating attack test set')

        for step in range(steps_per_epoch):
            # Reset attack noise variable
            sess.run(attack_reset)

            # Get one batch images & labels as inputs
            _, org_imgs, _, org_labs, _ = sess.run(attack_get_inputs, feed_dict=aa_init_feed)

            if run_benign:
                acc_1, acc_5, preds, labs, step_loss, step_regloss = sess.run(model.eval_op, feed_dict={is_training: bn_mode,
                                                                               attack.aa_init_on: True})
            else:
                # (Iteratively) Run attack
                for iter in range(attack_steps):
                    sess.run(attack_op, feed_dict=aa_on_feed)

                acc_1, acc_5, preds, labs, step_loss, step_regloss = sess.run(model.eval_op, feed_dict={is_training: bn_mode,
                                                                               attack.aa_init_on: False})

            if verbose:
                _progress_bar(step, steps_per_epoch)

            if params['Attack_config']['target_on']:
                real_labs = sess.run(attack.lab_org)
                print('\n     Real labels: ', real_labs)

            acc1_all.append(acc_1)
            acc5_all.append(acc_5)
            losses.append(step_loss)

        acc1_avg = np.mean(acc1_all)
        acc5_avg = np.mean(acc5_all)
        loss_avg = np.mean(losses)

        if print_acc:
            print('\n\x1b[0;30m' + '[Top1] =' + '\x1b[0m', '\x1b[4;30m' + '%.2f' % acc1_avg + ' %' + '\x1b[0m', end='\n')

        return acc1_avg, acc5_avg, loss_avg


def evaluate_attack(sess, model, attack, aa_ops, ds, params, is_training, run_benign=True, run_tr_mode=False, run_trainset=True, run_aa_eval=False):

    if run_tr_mode:
        if run_benign:
            if run_trainset:
                tr_bng_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TRAIN_SET, ds, is_training,
                                                                  bn_mode=BN_TRAIN_MODE, print_acc=False, run_benign=True, verbose=True)
                print('\n>> On TRAIN SET in TRAIN_MODE [Benign]: acc={:6.2f}%'.format(tr_bng_acc))

            val_bng_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.VALID_SET, ds, is_training,
                                                              bn_mode=BN_TRAIN_MODE, print_acc=False, run_benign=True, verbose=True)
            print('\n>> On VALID SET in TRAIN_MODE [Benign]: acc={:6.2f}%'.format(val_bng_acc))

            te_bng_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TEST_SET, ds, is_training,
                                                              bn_mode=BN_TRAIN_MODE, print_acc=False, run_benign=True, verbose=True)
            print('\n>> On TEST SET in TRAIN_MODE [Benign]: acc={:6.2f}%'.format(te_bng_acc))

        if run_trainset:
            tr_aa_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TRAIN_SET, ds, is_training,
                                                             bn_mode=BN_TRAIN_MODE, print_acc=False, run_benign=False, run_aa_eval=run_aa_eval, verbose=True)
            print('\n>> On TRAIN SET in TRAIN_MODE [Attack]: acc={:6.2f}%'.format(tr_aa_acc))

        val_aa_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.VALID_SET, ds, is_training,
                                                         bn_mode=BN_TRAIN_MODE, print_acc=False, run_benign=False, run_aa_eval=run_aa_eval, verbose=True)
        print('\n>> On VALID SET in TRAIN_MODE [Attack]: acc={:6.2f}%'.format(val_aa_acc))

        te_aa_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TEST_SET, ds, is_training,
                                                         bn_mode=BN_TRAIN_MODE, print_acc=False, run_benign=False, run_aa_eval=run_aa_eval, verbose=True)
        print('\n>> On TEST SET in TRAIN_MODE [Attack]: acc={:6.2f}%'.format(te_aa_acc))


    if run_benign:
        if run_trainset:
            tr_bng_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TRAIN_SET, ds, is_training,
                                                              bn_mode=BN_PERCEPTION_MODE, print_acc=False, run_benign=True, verbose=True)
            print('\n>>On TRAIN SET in EVAL_MODE [Benign]: acc={:6.2f}%'.format(tr_bng_acc))

        val_bng_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.VALID_SET, ds, is_training,
                                                          bn_mode=BN_PERCEPTION_MODE, print_acc=False, run_benign=True, verbose=True)
        print('\n>>On VALID SET in EVAL_MODE [Benign]: acc={:6.2f}%'.format(val_bng_acc))

        te_bng_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TEST_SET, ds, is_training,
                                                          bn_mode=BN_PERCEPTION_MODE, print_acc=False, run_benign=True, verbose=True)
        print('\n>>On TEST SET in EVAL_MODE [Benign]: acc={:6.2f}%'.format(te_bng_acc))

    if run_trainset:
        tr_aa_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TRAIN_SET, ds, is_training,
                                                         bn_mode=BN_PERCEPTION_MODE, print_acc=False, run_aa_eval=run_aa_eval, verbose=True)
        print('\n>> On TRAIN SET in EVAL_MODE [Attack]: acc={:6.2f}%'.format(tr_aa_acc))

    val_aa_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.VALID_SET, ds, is_training,
                                                     bn_mode=BN_PERCEPTION_MODE, print_acc=False, run_aa_eval=run_aa_eval, verbose=True)
    print('\n>>On VALID SET in EVAL_MODE [Attack]: acc={:6.2f}%'.format(val_aa_acc))

    te_aa_acc, _, _ = eval_classification_attack(sess, model, attack, aa_ops, params, ds.TEST_SET, ds, is_training,
                                                     bn_mode=BN_PERCEPTION_MODE, print_acc=False, run_aa_eval=run_aa_eval, verbose=True)
    print('\n>> On TEST SET in EVAL_MODE [Attack]: acc={:6.2f}%'.format(te_aa_acc))

