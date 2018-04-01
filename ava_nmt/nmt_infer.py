#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import argparse
import os
from shutil import copyfile

import numpy
import tensorflow as tf

import config_example
from build_inputs import build_source_char_inputs
from build_model import build_attention_model
from sequencing import TIME_MAJOR, MODE, optimistic_restore

def infer(src_vocab, src_data_file, trg_vocab,
          params, beam_size=1, batch_size=1, max_step=100,
          output_file='test.out', model_dir='models/'):

    save_output_dir = 'dev_outputs/'
    if not os.path.exists(save_output_dir):
        os.makedirs(save_output_dir)

    # ------------------------------------
    # prepare data
    # trg_data_file may be empty.
    # ------------------------------------

    # load parallel data
    parallel_data_generator = \
        build_source_char_inputs(src_vocab, src_data_file,
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
    source_sample_matrix = tf.placeholder(tf.float32, shape=(None, None, None),
                                       name='source_sample_matrix')
    source_word_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='source_word_seq_length')

    target_ids = None
    target_seq_length = None

    source_placeholders = {'src': source_ids,
                          'src_len': source_seq_length,
                          'src_sample_matrix':source_sample_matrix,
                          'src_word_len': source_word_seq_length}
    target_placeholders = {'trg': target_ids,
                          'trg_len': target_seq_length}


    decoder_output_eval, decoder_final_state = \
        build_attention_model(params, src_vocab, trg_vocab,
                              source_placeholders,
                              target_placeholders,
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

        output_file_name = os.path.join(save_output_dir,
                                output_file + last_ckpt.split('-')[-1])
        output_ = open(output_file_name, 'w')

        for step, current_input in enumerate(parallel_data_generator):
            current_input_dict = current_input._asdict()
            feed_dict = {}
            for key in source_placeholders.keys():
                feed_dict[source_placeholders[key]] = current_input_dict[key]

            # beam_ids_np: [seq_len, beam_size]
            # predicted_ids_np: [seq_len, beam_size]
            predicted_ids_np, beam_ids_np, log_probs_np = sess.run(
                [decoder_output_eval.predicted_ids,
                 decoder_output_eval.beam_ids,
                 decoder_final_state.log_probs],
                feed_dict=feed_dict)

            src_len_np = current_input_dict['src_len']
            data_batch_size = len(src_len_np)

            gathered_pred_ids = numpy.zeros_like(beam_ids_np)
            for idx in range(beam_ids_np.shape[0]):
                gathered_pred_ids = gathered_pred_ids[:, beam_ids_np[idx] %
                                                         beam_ids_np.shape[1]]
                gathered_pred_ids[idx, :] = predicted_ids_np[idx]

            seq_lens = []
            for idx in range(beam_ids_np.shape[1]):
                pred_ids_list = gathered_pred_ids[:, idx].tolist()
                seq_lens.append(pred_ids_list.index(trg_vocab.eos_id) + 1 \
                                    if trg_vocab.eos_id in pred_ids_list \
                                    else len(pred_ids_list))

            log_probs_np = log_probs_np / numpy.array(seq_lens)
            log_probs_np_list = numpy.split(log_probs_np, data_batch_size,
                                            axis=0)
            each_max_idx = [numpy.argmax(log_prob) + b * beam_size for
                            b, log_prob in enumerate(log_probs_np_list)]

            pids = gathered_pred_ids[:, each_max_idx]

            for b in range(data_batch_size):
                p = trg_vocab.id_to_token(pids[:, b].tolist())
                if TIME_MAJOR:
                    s = src_vocab.id_to_token(current_input_dict['src'][:, b].tolist())
                else:
                    s = src_vocab.id_to_token(current_input_dict['src'][b, :].tolist())
                print('src:', s)
                print('prd:', p)
                print('---------------------------')
                print('\n')
                output_.write(p + '\n')
            output_.flush()
        output_.close()
        copyfile(output_file_name, output_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    all_configs = [i for i in dir(config_example) if i.startswith('config_')]

    parser = argparse.ArgumentParser(description='Sequencing Training ...')
    parser.add_argument('--config', choices=all_configs,
                        help='specific config name, like {}, '
                             'see config.py'.format(all_configs),
                        required=True)
    parser.add_argument('--test-src', type=str,
                        help='test src file')
    parser.add_argument('--output-file', type=str,
                        help='test output file',
                        default='test.out')

    args = parser.parse_args()
    training_configs = getattr(config_example, args.config)()

    test_src_file = args.test_src if args.test_src else training_configs.test_src_file

    output_file = args.output_file

    infer(training_configs.src_vocab, test_src_file,
          training_configs.trg_vocab,
          training_configs.params,
          beam_size=training_configs.beam_size,
          batch_size=training_configs.batch_size,
          max_step=training_configs.max_step,
          model_dir=training_configs.model_dir,
          output_file=output_file)
