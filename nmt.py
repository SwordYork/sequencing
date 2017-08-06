#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import os
import time

import numpy
import tensorflow as tf
from build_inputs import build_vocab, build_parallel_inputs
from build_model import build_attention_model, optimistic_restore
from sequencing import MODE, TIME_MAJOR


def train(src_vocab_file, src_embedding_dim, src_data_file,
          trg_vocab_file, trg_embedding_dim, trg_data_file,
          params, train_step=200000, lr_rate=0.0005, batch_size=1,
          check_every_step=500, model_dir='models/'):
    # ------------------------------------
    # prepare data
    # ------------------------------------

    # load vocab
    src_vocab = build_vocab(src_vocab_file, src_embedding_dim, ' ')
    trg_vocab = build_vocab(trg_vocab_file, trg_embedding_dim, '')

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

    # attention model for training
    decoder_output, losses = build_attention_model(params, src_vocab, trg_vocab,
                                                   source_ids,
                                                   source_seq_length,
                                                   target_ids,
                                                   target_seq_length,
                                                   mode=MODE.TRAIN)

    # attention model for evaluating
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        decoder_output_eval, _ = build_attention_model(params, src_vocab,
                                                       trg_vocab,
                                                       source_ids,
                                                       source_seq_length,
                                                       target_ids,
                                                       target_seq_length,
                                                       mode=MODE.EVAL)

    # Calculate the average log perplexity in each batch
    loss_avg = tf.reduce_sum(losses) / tf.to_float(
        tf.reduce_sum(target_seq_length))

    # record loss curve
    tf.summary.scalar('loss', loss_avg)

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss_avg))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    # Create a saver object which will save all the variables
    saver = tf.train.Saver(max_to_keep=3)

    # GPU config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # ------------------------------------
    # training
    # ------------------------------------

    with tf.Session(config=config) as sess:
        # init
        init = tf.global_variables_initializer()
        global_step = 0
        model_path = os.path.join(model_dir, 'model.ckpt')
        last_ckpt = tf.train.latest_checkpoint(model_dir)

        # Merge all the summaries and write them out
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        if last_ckpt:
            global_step = int(last_ckpt.split('-')[-1])
            saver.restore(sess, last_ckpt)
        else:
            sess.run(init)

        # start training
        start_time = time.time()
        for step in range(train_step):
            src_np, src_len_np, trg_np, trg_len_np = next(
                parallel_data_generator)

            _, loss_avg_np, summary = \
                sess.run([train_op, loss_avg, merged],
                         feed_dict={source_ids: src_np,
                                    source_seq_length: src_len_np,
                                    target_ids: trg_np,
                                    target_seq_length: trg_len_np})
            train_writer.add_summary(summary, global_step)

            if step % check_every_step == 0:
                tf.logging.info('start_time: {}, {} steps / sec'.format(
                    start_time, check_every_step / (time.time() - start_time)))
                tf.logging.info('step: {}, loss: {}'.format(step, loss_avg_np))
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
                        tids = trg_np[:, i].tolist()
                    else:
                        tids = trg_np[i, :].tolist()
                    print(trg_vocab.id_to_token(pids))
                    print(trg_vocab.id_to_token(tids))
                    print('----------------------------------')

            global_step += 1


def infer(src_vocab_file, src_embedding_dim, src_data_file,
          trg_vocab_file, trg_embedding_dim, trg_data_file,
          params, output_file, beam_size=1, batch_size=1,
          model_dir='models/'):
    # ------------------------------------
    # prepare data
    # trg_data_file may be empty.
    # ------------------------------------

    # load vocab
    src_vocab = build_vocab(src_vocab_file, src_embedding_dim, ' ')
    trg_vocab = build_vocab(trg_vocab_file, trg_embedding_dim, '')

    # load parallel data
    parallel_data_generator = \
        build_parallel_inputs(src_vocab, trg_vocab,
                              src_data_file, trg_data_file,
                              batch_size=batch_size, buffer_size=96,
                              mode=MODE.INFER)

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

    decoder_output_eval, decoder_final_state = \
        build_attention_model(params, src_vocab, trg_vocab, source_ids,
                              source_seq_length, target_ids, target_seq_length,
                              beam_size=beam_size, mode=MODE.INFER,
                              max_step=100)

    # GPU config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        last_ckpt = tf.train.latest_checkpoint(model_dir)
        if last_ckpt:
            optimistic_restore(sess, last_ckpt)
        else:
            raise Exception('No checkpoint found ...')

        output_ = open(output_file, 'w')
        for step, curr_data in enumerate(parallel_data_generator):
            src_np, src_len_np, trg_np, trg_len_np = curr_data
            # beam_ids_np: [seq_len, beam_size]
            # predicted_ids_np: [seq_len, beam_size]
            predicted_ids_np, beam_ids_np, log_probs_np = sess.run(
                [decoder_output_eval.predicted_ids,
                 decoder_output_eval.beam_ids,
                 decoder_final_state.log_probs],
                feed_dict={source_ids: src_np,
                           source_seq_length: src_len_np,
                           target_ids: trg_np,
                           target_seq_length: trg_len_np})

            data_batch_size = len(src_len_np)

            gathered_pred_ids = numpy.zeros_like(beam_ids_np)
            for idx in range(beam_ids_np.shape[0]):
                gathered_pred_ids = gathered_pred_ids[:,
                                    beam_ids_np[idx] % beam_ids_np.shape[1]]
                gathered_pred_ids[idx, :] = predicted_ids_np[idx]

            seq_lens = []
            for idx in range(beam_ids_np.shape[1]):
                pred_ids_list = gathered_pred_ids[:, idx].tolist()
                seq_lens.append(pred_ids_list.index(
                    trg_vocab.eos_id) + 1 if trg_vocab.eos_id in pred_ids_list else len(
                    pred_ids_list))

            log_probs_np = log_probs_np / numpy.array(seq_lens)
            log_probs_np_list = numpy.split(log_probs_np, data_batch_size,
                                            axis=0)
            each_max_idx = [numpy.argmax(log_prob) + b * beam_size for
                            b, log_prob in enumerate(log_probs_np_list)]

            pids = gathered_pred_ids[:, each_max_idx]

            for b in range(data_batch_size):
                p = trg_vocab.id_to_token(pids[:, b].tolist())
                tf.logging.info(p)
                output_.write(p + '\n')
            output_.flush()
        output_.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    params = {'encoder': {'rnn_cell': {'state_size': 1024,
                                       'cell_name': 'BasicLSTMCell',
                                       'num_layers': 1,
                                       'input_keep_prob': 1.0,
                                       'output_keep_prob': 1.0},
                          'attention_key_size': 512},
              'decoder': {'rnn_cell': {'cell_name': 'BasicLSTMCell',
                                       'state_size': 1024,
                                       'num_layers': 1,
                                       'input_keep_prob': 1.0,
                                       'output_keep_prob': 1.0},
                          'logits': {'input_keep_prob': 1.0}}}

    train('data_en2zh/vocab.en', 512,
          'data_en2zh/en.tok.shuf.filter',
          'data_en2zh/vocab.zh', 512,
          'data_en2zh/zh.tok.shuf.filter',
          params=params, train_step=200000, batch_size=128,
          model_dir='models')

    infer('data_en2zh/vocab.en', 512,
          'data_en2zh/test.en',
          'data_en2zh/vocab.zh', 512,
          'data_en2zh/test.zh',
          params, 'test.out', beam_size=5, batch_size=10,
          model_dir='models')
