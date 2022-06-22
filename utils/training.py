import os
import sys
import tensorflow as tf
from utils import evaluation
from utils.evaluation import BN_PERCEPTION_MODE, BN_TRAIN_MODE
from datetime import datetime
from dateutil.relativedelta import relativedelta


def epoch2iters(params, fine_tune=False, optimize_score=False, use_train_set=True):
    epoch_iter = int(params['Dataset']['num_train_samples'] / params['Meta']['batchsize'])

    warmup_iter = 0

    if fine_tune:
        max_iter = int(params['Prune_config']['fine_tuning_epochs'] * epoch_iter)
        if params['Prune_config']['fine_tuning_warmup']:
            warmup_iter = int(params['Prune_config']['fine_tuning_warmup'] * epoch_iter)

        test_iter = int(epoch_iter)
        checkpoint_iter = int(epoch_iter)
    elif optimize_score:
        if not use_train_set:
            epoch_iter = int(params['Dataset']['num_valid_samples'] / params['Meta']['batchsize'])
            max_iter = int(params['Prune_config']['score_optim_epochs'] * epoch_iter)
        else:
            max_iter = int(params['Prune_config']['score_finetune_epochs'] * epoch_iter)

        test_iter = int(max_iter)
        checkpoint_iter = int(epoch_iter)
    else:
        max_iter = int(params['Meta']['epochs'] * epoch_iter)
        test_iter = int(params['Meta']['test_iter'] * epoch_iter)
        checkpoint_iter = int(params['Meta']['checkpoint_iter'] * epoch_iter)

    max_iter = max_iter if max_iter >= 1 else 1
    test_iter = test_iter if test_iter >= 1 else 1
    checkpoint_iter = checkpoint_iter if checkpoint_iter >= 1 else 1
    warmup_iter = warmup_iter if warmup_iter >= 1 else 0

    return max_iter, test_iter, checkpoint_iter, epoch_iter, warmup_iter


def train_classification(sess, model, ds, params, tb_writer, is_training):
    max_iter, test_iter, checkpoint_iter, epoch_iter, _ = epoch2iters(params)

    saver = tf.train.Saver()
    best_acc = 0.0

    cwd = os.getcwd()
    output_dir = os.path.join(cwd, params['Meta']['output_folder'])

    start_t = datetime.now()

    for step in range(0, max_iter+1):

        summary, _, total_loss, reg_loss = sess.run([model.tb_merged, model.train_op, model.total_loss, model.reg_loss],
                                                     feed_dict={ds.ds_switch: ds.TRAIN_SET, is_training: True})

        # Update learning rate
        model.lr.set_learning_rate(step)
        cur_lr = sess.run(model.lr.update_op, feed_dict={model.lr.val_temp: model.lr._value})

        if step == 0:
            # Skip step == 0 for test
            continue

        if step % test_iter == 0:
            tb_writer.add_summary(summary, step)

        if step % checkpoint_iter == 0 or step == max_iter:
            epoch = step // epoch_iter

            if params['Meta']['eval']:
                tr_acc, _, tr_loss, tr_regloss = evaluation.eval_classification(sess, model.eval_op, params, ds.TRAIN_SET, ds, is_training,
                                                           bn_mode=evaluation.BN_PERCEPTION_MODE, print_acc=False)
                val_acc, _, _, _ = evaluation.eval_classification(sess, model.eval_op, params, ds.VALID_SET, ds, is_training,
                                                            bn_mode=evaluation.BN_PERCEPTION_MODE, print_acc=False)
                te_acc, _, te_loss, te_regloss = evaluation.eval_classification(sess, model.eval_op, params, ds.TEST_SET, ds, is_training,
                                                           bn_mode=evaluation.BN_PERCEPTION_MODE, print_acc=False)

                try:
                    log = open(os.path.join(params['Meta']['output_folder'], 'Train.txt'), 'a')
                except:
                    log = open(os.path.join(params['Meta']['output_folder'], 'Train.txt'), 'w')
                    log.write('.......................\n')
                    log.write('Epoch\tLearning_rate\tLoss\tTr_acc\tVal_acc\tTe_acc\n')

                epoch_print = '> epoch {:3}: lr={:.4f}  tr_loss={:.4f}  tr_regloss={:.4f}  ' \
                              'tr_acc={:6.2f}%  val_acc={:6.2f}%  te_acc={:6.2f}%'.format(
                        epoch, cur_lr, tr_loss, tr_regloss, tr_acc, val_acc, te_acc)

                checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_last.ckpt')
                best_file = os.path.join(output_dir, params['Meta']['output_name'] + '_best.ckpt')
                if te_acc > best_acc:
                    best_acc = te_acc
                    print(epoch_print + '  Update best checkpoint!')
                    log.write(epoch_print + '  Update best checkpoint!\n')
                    saver.save(sess, best_file)
                else:
                    print(epoch_print)
                    log.write(epoch_print + '\n')
                    saver.save(sess, checkpoint_file)

                log.close()

    end_t = datetime.now()
    tr_time_diff = relativedelta(end_t, start_t)

    try:
        log = open(os.path.join(params['Meta']['output_folder'], 'Train.txt'), 'a')
    except:
        log = open(os.path.join(params['Meta']['output_folder'], 'Train.txt'), 'w')
    time_print = 'Total time use for adversarial train: {h}h:{m}m:{s}s.'.format(h=tr_time_diff.hours, m=tr_time_diff.minutes, s=tr_time_diff.seconds)
    log.write(time_print)
    log.close()


def fine_tuning(sess, model, var_list, ds, params, tb_writer, is_training, best_acc=50.0):

    max_iter, test_iter, checkpoint_iter, epoch_iter, _ = epoch2iters(params, fine_tune=True)

    saver = tf.train.Saver(var_list)

    cwd = os.getcwd()
    output_dir = os.path.join(cwd, params['Meta']['output_folder'])

    start_t = datetime.now()

    for step in range(0, max_iter+1):
        summary, _, total_loss = sess.run([model.tb_merged, model.train_op, model.total_loss], feed_dict={ds.ds_switch: ds.TRAIN_SET, is_training: True})
        model.lr.set_learning_rate(step)
        cur_lr = sess.run(model.lr.update_op, feed_dict={model.lr.val_temp: model.lr._value})

        if step % test_iter == 0:
            tb_writer.add_summary(summary, step)

        if step % checkpoint_iter == 0 or step == max_iter:
            epoch = step // epoch_iter

            if params['Meta']['eval']:
                tr_acc, _, tr_loss, tr_regloss = evaluation.eval_classification(sess, model.eval_op, params, ds.TRAIN_SET, ds, is_training,
                                                           bn_mode=evaluation.BN_PERCEPTION_MODE, print_acc=False)

                val_acc, _, _, _ = evaluation.eval_classification(sess, model.eval_op, params, ds.VALID_SET, ds, is_training,
                                                            bn_mode=evaluation.BN_PERCEPTION_MODE, print_acc=False)

                te_acc, _, tr_loss, tr_regloss = evaluation.eval_classification(sess, model.eval_op, params, ds.TEST_SET, ds, is_training,
                                                           bn_mode=evaluation.BN_PERCEPTION_MODE, print_acc=False)
                epoch_print = '> epoch {:3}: lr={:.4f}  tr_loss={:.4f}  tr_regloss={:.4f}  ' \
                              'tr_acc={:6.2f}%  val_acc={:6.2f}%  te_acc={:6.2f}%'.format(
                    epoch, cur_lr, tr_loss, tr_regloss, tr_acc, val_acc, te_acc)

                try:
                    log = open(os.path.join(params['Meta']['output_folder'], 'Fine_Tune.txt'), 'a')
                except:
                    log = open(os.path.join(params['Meta']['output_folder'], 'Fine_Tune.txt'), 'w')
                    log.write('.......................\n')
                    log.write('Epoch\tLearning_rate\tLoss\tTr_acc\tVal_acc\tTe_acc\n')

                checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_finetune_last.ckpt')
                best_file = os.path.join(output_dir, params['Meta']['output_name'] + '_finetune_best.ckpt')

                if te_acc > best_acc:
                    best_acc = te_acc
                    print(epoch_print + '  Update best checkpoint!')
                    log.write(epoch_print + '  Update best checkpoint!' + '\n')
                    saver.save(sess, best_file)
                else:
                    print(epoch_print)
                    log.write(epoch_print + '\n')
                    saver.save(sess, checkpoint_file)

                log.close()

    end_t = datetime.now()
    tr_time_diff = relativedelta(end_t, start_t)

    try:
        log = open(os.path.join(params['Meta']['output_folder'], 'Fine_Tune.txt'), 'a')
    except:
        log = open(os.path.join(params['Meta']['output_folder'], 'Fine_Tune.txt'), 'w')
    time_print = 'Total time use for adversarial train: {h}h:{m}m:{s}s.'.format(h=tr_time_diff.hours, m=tr_time_diff.minutes, s=tr_time_diff.seconds)
    log.write(time_print)
    log.close()


def adv_train_classification(sess, model, attack, aa_ops, ds, params, tb_writer, is_training):

    max_iter, test_iter, checkpoint_iter, epoch_iter, _ = epoch2iters(params)

    ds_name = params['Dataset']['name']

    attack_steps = params['Attack_config']['steps']

    aa_train_mode = params['Attack_config']['train_mode']
    aa_early_stop = params['Attack_config']['early_stop']
    stop_overfit = aa_train_mode == 'fast' and aa_early_stop
    overfit_ct = 0

    saver = tf.train.Saver()
    best_acc = 0.0
    best_aa_acc = 0.0
    best_aa_eval_acc = 0.0

    cwd = os.getcwd()
    output_dir = os.path.join(cwd, params['Meta']['output_folder'])

    start_t = datetime.now()  # Timer

    for step in range(0, max_iter+1):

        if ds_name == 'ImageNet2012':
            sys.stdout.write('\r>> current step: %d/%d' % (step, max_iter))
            sys.stdout.flush()

        # Reset attack noise variable
        sess.run(aa_ops['reset_attack'])

        # Get one batch images & labels as inputs
        sess.run(aa_ops['get_inputs'], feed_dict={ds.get_switch(): ds.TRAIN_SET, attack.aa_on: False})

        # Run attack
        for _ in range(attack_steps):
            sess.run(aa_ops['get_attack'], feed_dict={is_training: BN_TRAIN_MODE, attack.aa_on: True, attack.aa_init_on: False})

        summary, _, total_loss, reg_loss = sess.run([model.tb_merged, model.train_op, model.total_loss, model.reg_loss],
                                                    feed_dict={ds.ds_switch: ds.TRAIN_SET,
                                                               is_training: BN_TRAIN_MODE,
                                                               attack.aa_init_on: False})
        model.lr.set_learning_rate(step)
        cur_lr = sess.run(model.lr.update_op, feed_dict={model.lr.val_temp: model.lr._value})

        if step == 0:
            # Skip step == 0 for test
            continue

        if step % test_iter == 0:
            tb_writer.add_summary(summary, step)

        if step % checkpoint_iter == 0 or step == max_iter:

            try:
                log = open(os.path.join(params['Meta']['output_folder'], 'Adv_Train.txt'), 'a')
            except:
                log = open(os.path.join(params['Meta']['output_folder'], 'Adv_Train.txt'), 'w')
                log.write('.......................\n')

            checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_adv_last.ckpt')
            best_file = os.path.join(output_dir, params['Meta']['output_name'] + '_adv_best.ckpt')

            epoch = step // epoch_iter

            if ds_name != 'ImageNet2012':

                bng_tr_acc, _, bng_tr_loss, bng_tr_regloss, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                         ds.TRAIN_SET, ds, is_training,
                                                                         bn_mode=BN_PERCEPTION_MODE,
                                                                         print_acc=False,
                                                                         run_benign=True)

                aa_tr_acc, _, aa_tr_loss, aa_tr_regloss, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                        ds.TRAIN_SET, ds, is_training,
                                                                        bn_mode=BN_PERCEPTION_MODE,
                                                                        print_acc=False)
            else:
                bng_tr_acc, aa_tr_acc = 0.0, 0.0

            bng_te_acc, _, bng_te_loss, bng_te_regloss, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                     ds.TEST_SET, ds, is_training,
                                                                     bn_mode=BN_PERCEPTION_MODE,
                                                                     print_acc=False,
                                                                     run_benign=True)

            aa_te_acc, _, aa_te_loss, aa_te_regloss, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                    ds.TEST_SET, ds, is_training,
                                                                    bn_mode=BN_PERCEPTION_MODE,
                                                                    print_acc=False)

            step_end_t = datetime.now()
            if step == 0:
                time_stamp = ' [epoch: --- min | rest: --- hour]'
            else:
                epoch_t = (step_end_t - start_t).total_seconds() / (60.0 * epoch)  # second -> minute
                rest_t = epoch_t / 60.0 * (max_iter/epoch_iter - epoch)
                time_stamp = ' [epoch: {:2.2f} min | rest: {:3.2f} hour]'.format(epoch_t, rest_t)

            if stop_overfit:
                assert 'get_attack_eval' in list(aa_ops.keys()), 'No eval. attack added in aa_ops!'
                aa_eval_te_acc, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                             ds.TEST_SET, ds, is_training,
                                                                             bn_mode=BN_PERCEPTION_MODE,
                                                                             print_acc=False,
                                                                             run_aa_eval=True)

                epoch_print = '> epoch {:3}: lr={:.4f}  loss={:.4f}  reg_loss={:.4f}  tr_acc={:6.2f}%  ' \
                              'te_acc={:6.2f}%  aa_tr_acc={:6.2f}%  aa_te_acc={:6.2f}%  aa_pgd_te_eval={:6.2f}%'.format(
                    epoch, cur_lr, total_loss, reg_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc, aa_eval_te_acc)

                if aa_eval_te_acc + 15.0 < best_aa_eval_acc:
                    overfit_ct += 1

                    # Test over-fitting
                    if overfit_ct == 5:
                        print(epoch_print + '  Overfitted! STOP!')
                        log.write(epoch_print + '  Overfitted! STOP!' + '\n')
                        log.close()
                        break
                elif aa_eval_te_acc > best_aa_eval_acc:
                    overfit_ct = 0
                    best_aa_eval_acc = aa_eval_te_acc
                    best_acc = bng_te_acc
                    print(epoch_print + '  Update best checkpoint!')
                    log.write(epoch_print + '  Update best checkpoint!' + '\n')
                    saver.save(sess, best_file)
                else:
                    print(epoch_print)
                    log.write(epoch_print + '\n')
                    saver.save(sess, checkpoint_file)

            else:
                epoch_print = '> epoch {:3}: lr={:.4f}  tr_aa_loss={:.4f}  tr_aa_regloss={:.4f}  te_aa_loss={:.4f}  tr_acc={:6.2f}%  ' \
                              'te_acc={:6.2f}%  aa_tr_acc={:6.2f}%  aa_te_acc={:6.2f}%'.format(
                    epoch, cur_lr, aa_tr_loss, aa_tr_regloss, aa_te_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)

                if aa_te_acc > best_aa_acc:
                    best_acc = bng_te_acc
                    best_aa_acc = aa_te_acc
                    print(epoch_print + time_stamp + '  Update best checkpoint!')
                    log.write(epoch_print + '  Update best checkpoint!\n')
                    saver.save(sess, best_file)
                else:
                    print(epoch_print + time_stamp)
                    log.write(epoch_print + '\n')
                    saver.save(sess, checkpoint_file)

            log.close()


    end_t = datetime.now()
    tr_time_diff = relativedelta(end_t, start_t)

    try:
        log = open(os.path.join(params['Meta']['output_folder'], 'Adv_Train.txt'), 'a')
    except:
        log = open(os.path.join(params['Meta']['output_folder'], 'Adv_Train.txt'), 'w')
    time_print = 'Total time use for adversarial train: {h}h:{m}m:{s}s.'.format(h=tr_time_diff.hours, m=tr_time_diff.minutes, s=tr_time_diff.seconds)
    log.write(time_print)
    log.close()


def adv_fine_tuning(sess, model, attack, aa_ops, var_list, ds, params, tb_writer, is_training, output_dir, opt_score=False):

    max_iter, test_iter, checkpoint_iter, epoch_iter, warmup_iter = epoch2iters(params, fine_tune=True, optimize_score=opt_score)

    saver = tf.train.Saver(var_list)

    attack_steps = params['Attack_config']['steps']

    # aa_train_mode = params['Attack_config']['train_mode']
    # aa_early_stop = params['Attack_config']['early_stop']
    # stop_overfit = aa_train_mode == 'fast' and aa_early_stop
    # overfit_ct = 0

    best_aa_acc = 0.0

    start_t = datetime.now()  # Timer

    print("\n=========== Start finetune with '{}' ===========\n".format(params['Attack_config']['train_mode']))

    try:
        log = open(os.path.join(output_dir, 'Adv_Fine_Tune.csv'), 'x')
    except:
        log = open(os.path.join(output_dir, 'Adv_Fine_Tune.csv'), 'w')
    log.write('Epoch\tLearning_rate\tLoss\tTr_acc\tTe_acc\tTr_aa_acc\tTe_aa_acc\n')
    log.close()

    for step in range(0, max_iter+1):

        # Get one batch images & labels as inputs
        sess.run(aa_ops['get_inputs'], feed_dict={ds.get_switch(): ds.TRAIN_SET, attack.aa_on: False})

        # Reset attack noise variable
        sess.run(aa_ops['reset_attack'])

        # Run attack
        if step >= warmup_iter:
            for iter in range(attack_steps):
                sess.run(aa_ops['get_attack'], feed_dict={is_training: BN_TRAIN_MODE,
                                                          attack.aa_on: True,
                                                          attack.aa_init_on: False})

        summary, _, total_loss = sess.run([model.tb_merged, model.train_op, model.total_loss],
                                          feed_dict={ds.ds_switch: ds.TRAIN_SET,
                                                     is_training: True,
                                                     attack.aa_init_on: False})
        model.lr.set_learning_rate(step)
        cur_lr = sess.run(model.lr.update_op, feed_dict={model.lr.val_temp: model.lr._value})

        if step % test_iter == 0 and tb_writer is not None:
            tb_writer.add_summary(summary, step)

        if step % checkpoint_iter == 0 or step == max_iter:
            epoch = step // epoch_iter
            # checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_finetune_%d.ckpt' % (step))
            # saver.save(sess, checkpoint_file)

            if params['Meta']['eval']:
                bng_tr_acc, _, _, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                         ds.TRAIN_SET, ds, is_training,
                                                                         bn_mode=BN_PERCEPTION_MODE,
                                                                         print_acc=False,
                                                                         run_benign=True)

                bng_te_acc, _, _, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                         ds.TEST_SET, ds, is_training,
                                                                         bn_mode=BN_PERCEPTION_MODE,
                                                                         print_acc=False,
                                                                         run_benign=True)

                aa_tr_acc, _, tr_aa_loss, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                        ds.TRAIN_SET, ds, is_training,
                                                                        bn_mode=BN_PERCEPTION_MODE,
                                                                        print_acc=False)
                                                                        # run_aa_eval=True)

                aa_te_acc, _, te_aa_loss, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                        ds.TEST_SET, ds, is_training,
                                                                        bn_mode=BN_PERCEPTION_MODE,
                                                                        print_acc=False)
                                                                        # run_aa_eval=True)

                epoch_print = \
                    '> epoch {:3}: lr={:.4f}  tr_aa_loss={:.4f} te_aa_loss={:.4f}  tr_acc={:6.2f}%  te_acc={:6.2f}%  ' \
                    'aa_tr_acc={:6.2f}%  aa_te_acc={:6.2f}%'.format(
                        epoch, cur_lr, tr_aa_loss, te_aa_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)

                checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_adv_finetune_last.ckpt')
                best_file = os.path.join(output_dir, params['Meta']['output_name'] + '_adv_finetune_best.ckpt')

                log = open(os.path.join(output_dir, 'Adv_Fine_Tune.csv'), 'a')

                if aa_te_acc > best_aa_acc:
                    best_aa_acc = aa_te_acc
                    print(epoch_print + '  Update best checkpoint!')
                    log.write(('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' % (
                        epoch, cur_lr, total_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)) + 'Update Best! \n')
                    saver.save(sess, best_file)
                else:
                    print(epoch_print)
                    log.write(('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' % (
                        epoch, cur_lr, total_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)) + '\n')
                    saver.save(sess, checkpoint_file)

                log.close()

    end_t = datetime.now()
    tr_time_diff = relativedelta(end_t, start_t)

    log = open(os.path.join(output_dir, 'Adv_Fine_Tune.csv'), 'a')
    time_print = 'Total time use for adversarial fine_tune: {h}h:{m}m:{s}s.'.format(h=tr_time_diff.hours, m=tr_time_diff.minutes, s=tr_time_diff.seconds)
    print(time_print)
    log.write(time_print)
    log.close()


def adv_score_opt(sess, model, attack, aa_ops, ds, params, is_training, var_list, output_dir, use_train_set=False, save_eval=False):

    max_iter, test_iter, checkpoint_iter, epoch_iter, _ = epoch2iters(params, optimize_score=True, use_train_set=use_train_set)

    attack_steps = params['Attack_config']['steps']

    saver = tf.train.Saver(var_list)

    best_aa_acc = 0.0

    start_t = datetime.now()  # Timer

    print("\n=========== Score optimize with '{}' ===========\n".format(params['Attack_config']['train_mode']))

    data_switch = ds.TRAIN_SET if use_train_set else ds.VALID_SET

    try:
        log = open(os.path.join(output_dir, 'Adv_Score_Optim.csv'), 'x')
    except:
        log = open(os.path.join(output_dir, 'Adv_Score_Optim.csv'), 'w')
    log.write('Epoch\tLearning_rate\tLoss\tTr_acc\tTe_acc\tTr_aa_acc\tTe_aa_acc\n')
    log.close()

    for step in range(0, max_iter+1):

        # Get one batch images & labels as inputs
        sess.run(aa_ops['get_inputs'], feed_dict={ds.get_switch(): data_switch, attack.aa_on: False})

        # Reset attack noise variable
        sess.run(aa_ops['reset_attack'])

        # Run attack
        for iter in range(attack_steps):
            sess.run(aa_ops['get_attack'], feed_dict={is_training: BN_TRAIN_MODE,
                                                      attack.aa_on: True,
                                                      attack.aa_init_on: False})

        summary, _, total_loss = sess.run([model.tb_merged, model.train_op, model.total_loss],
                                          feed_dict={ds.ds_switch: data_switch,
                                                     is_training: True,
                                                     attack.aa_init_on: False})
        model.lr.set_learning_rate(step)

        # if step % epoch_iter == 0:
        #     epoch = step // epoch_iter
        #     print('> Epoch [{:d}]: total_loss={:.4f}'.format(epoch, total_loss))

        cur_lr = sess.run(model.lr.update_op, feed_dict={model.lr.val_temp: model.lr._value})

        if save_eval:

            if step % checkpoint_iter == 0 or step == max_iter:
                epoch = step // epoch_iter
                # checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_finetune_%d.ckpt' % (step))
                # saver.save(sess, checkpoint_file)

                bng_tr_acc, _, _, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                         ds.TRAIN_SET, ds, is_training,
                                                                         bn_mode=BN_PERCEPTION_MODE,
                                                                         print_acc=False,
                                                                         run_benign=True)

                bng_te_acc, _, _, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                         ds.TEST_SET, ds, is_training,
                                                                         bn_mode=BN_PERCEPTION_MODE,
                                                                         print_acc=False,
                                                                         run_benign=True)

                aa_tr_acc, _, tr_aa_loss, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                        ds.TRAIN_SET, ds, is_training,
                                                                        bn_mode=BN_PERCEPTION_MODE,
                                                                        print_acc=False)

                aa_te_acc, _, te_aa_loss, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                        ds.TEST_SET, ds, is_training,
                                                                        bn_mode=BN_PERCEPTION_MODE,
                                                                        print_acc=False)

                epoch_print = \
                    '> epoch {:3}: lr={:.4f}  tr_aa_loss={:.4f} te_aa_loss={:.4f}  tr_acc={:6.2f}%  te_acc={:6.2f}%  ' \
                    'aa_tr_acc={:6.2f}%  aa_te_acc={:6.2f}%'.format(
                        epoch, cur_lr, tr_aa_loss, te_aa_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)

                checkpoint_file = os.path.join(output_dir, params['Meta']['output_name'] + '_adv_score_last.ckpt')
                best_file = os.path.join(output_dir, params['Meta']['output_name'] + '_adv_score_best.ckpt')

                log = open(os.path.join(output_dir, 'Adv_Score_Optim.csv'), 'a')

                if aa_te_acc > best_aa_acc:
                    best_aa_acc = aa_te_acc
                    print(epoch_print + '  Update best checkpoint!')
                    log.write(('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' % (
                        epoch, cur_lr, total_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)) + 'Update Best! \n')
                    saver.save(sess, best_file)
                else:
                    print(epoch_print)
                    log.write(('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' % (
                        epoch, cur_lr, total_loss, bng_tr_acc, bng_te_acc, aa_tr_acc, aa_te_acc)) + '\n')
                    saver.save(sess, checkpoint_file)

                log.close()

    end_t = datetime.now()
    tr_time_diff = relativedelta(end_t, start_t)

    log = open(os.path.join(output_dir, 'Adv_Score_Optim.csv'), 'a')
    time_print = 'Total time use for adversarial score optimize: {h}h:{m}m:{s}s.'.format(h=tr_time_diff.hours,
                                                                                         m=tr_time_diff.minutes,
                                                                                         s=tr_time_diff.seconds)
    print(time_print)
    log.write(time_print)
    log.close()


def _short_finetune(sess, model, attack, aa_ops, ds, params, is_training, use_train_set=False):

    max_iter, test_iter, checkpoint_iter, epoch_iter, _ = epoch2iters(params, optimize_score=True, use_train_set=use_train_set)

    attack_steps = params['Attack_config']['steps']

    num_epochs = params['Prune_config']['short_ft_epochs']

    print("\n=========== Short finetune with '{}' ===========\n".format(params['Attack_config']['train_mode']))

    data_switch = ds.TRAIN_SET if use_train_set else ds.VALID_SET

    for step in range(0, epoch_iter*num_epochs+1):

        # Get one batch images & labels as inputs
        sess.run(aa_ops['get_inputs'], feed_dict={ds.get_switch(): data_switch, attack.aa_on: False})

        # Reset attack noise variable
        sess.run(aa_ops['reset_attack'])

        # Run attack
        for iter in range(attack_steps):
            sess.run(aa_ops['get_attack'], feed_dict={is_training: BN_TRAIN_MODE,
                                                      attack.aa_on: True,
                                                      attack.aa_init_on: False})

        summary, _, total_loss = sess.run([model.tb_merged, model.train_op, model.total_loss],
                                          feed_dict={ds.ds_switch: data_switch,
                                                     is_training: True,
                                                     attack.aa_init_on: False})
        model.lr.set_learning_rate(step)

        # if step % epoch_iter == 0:
        #     epoch = step // epoch_iter
        #     print('> Epoch [{:d}]: total_loss={:.4f}'.format(epoch, total_loss))

        cur_lr = sess.run(model.lr.update_op, feed_dict={model.lr.val_temp: model.lr._value})

        if step % checkpoint_iter == 0 or step == num_epochs*epoch_iter:
            epoch = step // epoch_iter

            bng_val_acc, _, _, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                     ds.VALID_SET, ds, is_training,
                                                                     bn_mode=BN_PERCEPTION_MODE,
                                                                     print_acc=False,
                                                                     run_benign=True)

            aa_val_acc, _, _, _, _ = evaluation.eval_classification_attack(sess, model, attack, aa_ops, params,
                                                                    ds.VALID_SET, ds, is_training,
                                                                    bn_mode=BN_PERCEPTION_MODE,
                                                                    print_acc=False)

            print('> epoch {:3}: bng_acc={:6.2f}%  adv_acc={:6.2f}%'.format(epoch, bng_val_acc, aa_val_acc))
