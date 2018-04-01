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
import numpy

import tensorflow as tf

import config_example
from build_inputs import build_parallel_char_inputs
from build_model import build_attention_model
from sequencing import MODE, TIME_MAJOR, optimistic_restore


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
        build_parallel_char_inputs(src_vocab, trg_vocab,
                              src_data_file, trg_data_file,
                              batch_size=batch_size, buffer_size=36,
                              mode=MODE.TRAIN)

    # ------------------------------------
    # build model
    # ------------------------------------

    # placeholder
    source_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='source_ids')
    source_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='source_seq_length')
    source_sample_matrix = tf.placeholder(tf.float32, shape=(None, None, None),
                                       name='source_sample_matrix')
    source_word_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='source_word_seq_length')

    target_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='target_ids')
    target_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='target_seq_length')

    source_placeholders = {'src': source_ids,
                          'src_len': source_seq_length,
                          'src_sample_matrix':source_sample_matrix,
                          'src_word_len': source_word_seq_length}
    target_placeholders = {'trg': target_ids,
                          'trg_len': target_seq_length}



    # Creates a variable to hold the global_step.
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # attention model for training
    _, total_loss_avg, entropy_loss_avg, reward_loss_rmse, reward_predicted = \
        build_attention_model(params, src_vocab, trg_vocab,
                              source_placeholders,
                              target_placeholders,
                              mode=mode,
                              burn_in_step=burn_in_step,
                              increment_step=increment_step,
                              max_step=max_step)

    # attention model for evaluating
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        decoder_output_eval, _ = \
            build_attention_model(params, src_vocab, trg_vocab,
                                  source_placeholders,
                                  target_placeholders,
                                  mode=MODE.EVAL,
                                  max_step=max_step)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.001)

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
    config.gpu_options.allow_growth = False

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
            current_input = next(parallel_data_generator)
            current_input_dict = current_input._asdict()
            feed_dict = {}
            for key in source_placeholders.keys():
                feed_dict[source_placeholders[key]] = current_input_dict[key]
            for key in target_placeholders.keys():
                feed_dict[target_placeholders[key]] = current_input_dict[key]

            _, total_loss_avg_np, summary, gradients_norm_np, gradients_np, reward_predicted_np, global_step = \
                sess.run([train_op, total_loss_avg, summary_merged, gradients_norm, gradients,
                          reward_predicted, global_step_tensor],
                         feed_dict=feed_dict)
            train_writer.add_summary(summary, global_step)

            if numpy.isnan(gradients_norm_np):
                print(gradients_norm_np, gradients_np)
                break

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
                             feed_dict=feed_dict)

                # print eval results
                for i in range(10):
                    pids = predicted_ids_np[:, i].tolist()
                    if TIME_MAJOR:
                        print(TIME_MAJOR)
                        print("shape",predicted_ids_np.shape)
                        print("pid is", pids)
                        sids = current_input_dict['src'][:, i].tolist()
                        tids = current_input_dict['trg'][:, i].tolist()
                    else:
                        sids = current_input_dict['src'][i, :].tolist()
                        tids = current_input_dict['trg'][i, :].tolist()
                    print('src:', src_vocab.id_to_token(sids))
                    print('prd:', trg_vocab.id_to_token(pids))
                    print('trg:', trg_vocab.id_to_token(tids))
                    print('---------------------------------')
                print('---------------------------------')
                print('---------------------------------')
                print('\n')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    all_configs = [i for i in dir(config_example) if i.startswith('config_')]

    parser = argparse.ArgumentParser(description='Sequencing Training ...')
    parser.add_argument('--config', choices=all_configs,
                        help='specific config name, like {}, '
                             'see config.py'.format(all_configs),
                        required=True)
    parser.add_argument('--mode', choices=['train', 'rl'], default='train')

    args = parser.parse_args()

    training_configs = getattr(config_example, args.config)()
    print("training_configs.params is", training_configs.params)
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
