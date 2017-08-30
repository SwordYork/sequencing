#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import argparse
import os
import time
from datetime import datetime

import tensorflow as tf

import config
from build_inputs import build_parallel_inputs
from build_model import build_attention_model, optimistic_restore
from sequencing import MODE, TIME_MAJOR


def train(src_vocab, src_data_file, trg_vocab, trg_data_file,
          params, batch_size=1, max_step=300, train_steps=200000,
          lr_rate=0.0005, clip_gradient_norm=5., check_every_step=500,
          model_dir='models/', burn_in_step=500, increment_step=1000,
          mode=MODE.TRAIN):
    # ------------------------------------
    # prepare data
    # ------------------------------------

    # load parallel data
    parallel_data_generator = \
        build_parallel_inputs(src_vocab, trg_vocab,
                              src_data_file, trg_data_file,
                              batch_size=batch_size, buffer_size=96,
                              mode=MODE.TRAIN)

    # ------------------------------------
    # build model
    # ------------------------------------

    # placeholder
    source_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='source_ids')
    source_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='source_seq_length')

    target_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='target_ids')
    target_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='target_seq_length')

    # Creates a variable to hold the global_step.
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # attention model for training
    _, total_loss_avg, entropy_loss_avg, reward_loss_rmse, reward_predicted = \
        build_attention_model(params, src_vocab, trg_vocab, source_ids,
                              source_seq_length, target_ids,
                              target_seq_length, mode=mode,
                              burn_in_step=burn_in_step,
                              increment_step=increment_step,
                              max_step=max_step)

    # attention model for evaluating
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        decoder_output_eval, _ = \
            build_attention_model(params, src_vocab, trg_vocab, source_ids,
                                  source_seq_length, target_ids,
                                  target_seq_length, mode=MODE.EVAL,
                                  max_step=max_step)

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr_rate)

    gradients, variables = zip(*optimizer.compute_gradients(total_loss_avg))
    gradients_norm = tf.global_norm(gradients)

    gradients, _ = tf.clip_by_global_norm(gradients, clip_gradient_norm,
                                          use_norm=gradients_norm)
    train_op = optimizer.apply_gradients(zip(gradients, variables),
                                         global_step=global_step_tensor)

    # record loss curve
    tf.summary.scalar('total_loss', total_loss_avg)
    tf.summary.scalar('entropy_loss_avg', entropy_loss_avg)
    tf.summary.scalar('reward_predicted', reward_predicted)
    tf.summary.scalar('reward_loss_rmse', reward_loss_rmse)
    tf.summary.scalar('gradients_norm', gradients_norm)

    # Create a saver object which will save all the variables
    saver_var_list = tf.trainable_variables()
    saver_var_list.append(global_step_tensor)
    saver = tf.train.Saver(var_list=saver_var_list, max_to_keep=3)

    # GPU config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # ------------------------------------
    # training
    # ------------------------------------

    with tf.Session(config=config) as sess:
        # init
        init = tf.global_variables_initializer()
        sess.run(init)

        model_path = os.path.join(model_dir, 'model.ckpt')
        last_ckpt = tf.train.latest_checkpoint(model_dir)

        # Merge all the summaries and write them out
        summary_merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        if last_ckpt:
            optimistic_restore(sess, last_ckpt)

        tf.logging.info('Train model ...')

        # start training
        start_time = time.time()
        for step in range(1, train_steps):
            src_np, src_len_np, trg_np, trg_len_np = next(
                parallel_data_generator)
            _, total_loss_avg_np, summary, reward_predicted_np, global_step = \
                sess.run([train_op, total_loss_avg, summary_merged,
                          reward_predicted, global_step_tensor],
                         feed_dict={source_ids: src_np,
                                    source_seq_length: src_len_np,
                                    target_ids: trg_np,
                                    target_seq_length: trg_len_np})
            train_writer.add_summary(summary, global_step)

            if step % check_every_step == 0:
                tf.logging.info('start_time: {}, {} steps / sec'.format(
                    datetime.fromtimestamp(start_time).strftime('%Y-%m-%d '
                                                                '%H:%M:%S'),
                    check_every_step / (time.time() - start_time)))
                tf.logging.info(
                    'global_step: {}, step: {}, total_loss: {}'.format(
                        global_step, step, total_loss_avg_np))
                start_time = time.time()

                saver.save(sess, model_path, global_step=global_step)
                predicted_ids_np = \
                    sess.run(decoder_output_eval.predicted_ids,
                             feed_dict={source_ids: src_np,
                                        source_seq_length: src_len_np,
                                        target_ids: trg_np,
                                        target_seq_length: trg_len_np})

                # print eval results
                for i in range(10):
                    pids = predicted_ids_np[:, i].tolist()
                    if TIME_MAJOR:
                        sids = src_np[:, i].tolist()
                        tids = trg_np[:, i].tolist()
                    else:
                        sids = src_np[i, :].tolist()
                        tids = trg_np[i, :].tolist()
                    print('src:', src_vocab.id_to_token(sids))
                    print('prd:', trg_vocab.id_to_token(pids))
                    print('trg:', trg_vocab.id_to_token(tids))
                    print('---------------------------------')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    all_configs = [i for i in dir(config) if i.startswith('config_')]

    parser = argparse.ArgumentParser(description='Sequencing Training ...')
    parser.add_argument('--config', choices=all_configs,
                        help='specific config name, like {}, '
                             'see config.py'.format(all_configs),
                        required=True)
    parser.add_argument('--mode', choices=['train', 'rl'], default='train')

    args = parser.parse_args()

    training_configs = getattr(config, args.config)()

    if args.mode == 'rl':
        mode = MODE.RL
    else:
        mode = MODE.TRAIN

    train(training_configs.src_vocab, training_configs.train_src_file,
          training_configs.trg_vocab, training_configs.train_trg_file,
          params=training_configs.params,
          batch_size=training_configs.batch_size,
          max_step=training_configs.max_step,
          train_steps=training_configs.train_steps,
          lr_rate=training_configs.lr_rate,
          clip_gradient_norm=training_configs.clip_gradient_norm,
          model_dir=training_configs.model_dir,
          burn_in_step=training_configs.burn_in_step,
          increment_step=training_configs.increment_step,
          mode=mode)
