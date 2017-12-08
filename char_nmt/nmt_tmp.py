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
from build_inputs import build_parallel_char_inputs
from build_model import build_attention_model
from sequencing import MODE, TIME_MAJOR, optimistic_restore


def train(src_vocab, src_data_file, trg_vocab, trg_data_file,
          params, batch_size=1, max_step=300, train_steps=200000,
          lr_rate=0.0005, clip_gradient_norm=5., check_every_step=100,
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

    target_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='target_ids')
    target_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='target_seq_length')

    source_placeholders = {'src': source_ids,
                          'src_len': source_seq_length,
                          'src_sample_matrix': source_sample_matrix}
    target_placeholders = {'trg': target_ids,
                          'trg_len': target_seq_length}



    # Creates a variable to hold the global_step.
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # attention model for training
    tmp = \
        build_attention_model(params, src_vocab, trg_vocab, 
                              source_placeholders,
                              target_placeholders,
                              mode=mode,
                              burn_in_step=burn_in_step,
                              increment_step=increment_step,
                              max_step=max_step)

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

            tmp_np, global_step = \
                sess.run([tmp, global_step_tensor],
                         feed_dict=feed_dict)
            print(global_step, tmp_np.shape)


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
