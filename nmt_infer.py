#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import numpy
import tensorflow as tf

from build_inputs import build_parallel_inputs
from build_model import build_attention_model, optimistic_restore
from config import get_config
from sequencing import MODE


def infer(src_vocab, src_data_file, trg_vocab, trg_data_file,
          params, beam_size=1, batch_size=1, max_step=100,
          output_file='test.out', model_dir='models/'):
    # ------------------------------------
    # prepare data
    # trg_data_file may be empty.
    # ------------------------------------

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
                              max_step=max_step)

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
    configs = get_config('word2pos')

    infer(configs.src_vocab, configs.test_src_file,
          configs.trg_vocab, configs.test_trg_file,
          configs.params,
          beam_size=configs.beam_size,
          batch_size=configs.batch_size,
          max_step=configs.max_step,
          model_dir=configs.model_dir)
