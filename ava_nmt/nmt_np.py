#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import sys

import sequencing_np as sqn
from config import get_config
from sequencing_np import MODE, np, DTYPE, TIME_MAJOR


def build_vocab(vocab_file, embedding_dim, delimiter=' '):
    # construct vocab
    with open(vocab_file, 'r') as f:
        symbols = [s[:-1] for s in f.readlines()]
    vocab = sqn.Vocab(symbols, embedding_dim, delimiter)
    return vocab


def build_attention_model_np(params, dict_var_vals, src_vocab, trg_vocab,
                             source_ids, source_seq_length, beam_size=1,
                             max_step=100):
    """
    Build the model.

    :param params: see `sequencing`.
    :param dict_var_vals: numpy array of trained variables
    :param src_vocab:
    :param trg_vocab:
    :param source_ids:
    :param source_seq_length:
    :param beam_size:
    :param max_step:
    :return:
    """

    mode = MODE.INFER
    batch_size = 1

    graph = sqn.Graph()
    # Because source encoder is different to the target feedback,
    # we construct source_embedding_table manually
    source_embedding_table = sqn.LookUpOp(name='source')

    state_size = params['encoder']['rnn_cell']['state_size']
    init_states = []
    if params['encoder']['rnn_cell']['cell_name'] != 'BasicLSTMCell':
        init_states.append(np.zeros((batch_size, state_size),
                                    dtype=DTYPE))
        init_states.append(np.zeros((batch_size, state_size),
                                    dtype=DTYPE))
    else:
        init_states.append(
            (np.zeros((batch_size, state_size), dtype=DTYPE),) * 2)
        init_states.append(
            (np.zeros((batch_size, state_size), dtype=DTYPE),) * 2)

    encoder = sqn.StackBidirectionalRNNEncoder(params['encoder'],
                                               init_states=init_states,
                                               name='stack_rnn')
    # initialize encoder first
    graph.initialize(dict_var_vals)
    source_embedded = source_embedding_table(source_ids)
    encoded_representation = encoder.encode(source_embedded, source_seq_length)

    attention_keys = encoded_representation.attention_keys
    attention_values = encoded_representation.attention_values
    attention_length = encoded_representation.attention_length

    # feedback
    feedback = sqn.BeamFeedBack(trg_vocab, beam_size, max_step, name='feedback')

    # attention
    attention = sqn.Attention(attention_keys, attention_values,
                              attention_length)

    init_states = []
    if params['decoder']['rnn_cell']['cell_name'] != 'BasicLSTMCell':
        init_states.append(np.zeros((batch_size * beam_size, state_size),
                                    dtype=DTYPE))
        init_states.append(np.zeros((batch_size * beam_size, state_size),
                                    dtype=DTYPE))
    else:
        init_states.append(
            (np.zeros((batch_size * beam_size, state_size), dtype=DTYPE),) * 2)
        init_states.append(
            (np.zeros((batch_size * beam_size, state_size), dtype=DTYPE),) * 2)
    # decoder
    decoder = sqn.AttentionRNNDecoder(params['decoder'], attention,
                                      feedback, init_states=init_states,
                                      mode=mode, name='attention_decoder')
    # initialize decoder
    graph.initialize(dict_var_vals)

    decoder_output, decoder_final_state = sqn.decode_loop(decoder)

    return decoder_output, decoder_final_state


def infer(src_vocab, trg_vocab, src_sentence, params, beam_size=1,
          model_dir='models/'):
    # ------------------------------------
    # prepare data
    # ------------------------------------

    # load parallel data
    source_ids = np.asarray([src_vocab.string_to_ids(src_sentence)],
                            dtype=np.int32)
    if TIME_MAJOR:
        source_ids = source_ids.T

    source_seq_length = np.asarray([len(source_ids)], dtype=np.int32)

    # ------------------------------------
    # build model
    # ------------------------------------

    dict_var_vals = np.load(model_dir + 'model.ckpt.npz')
    decoder_output_eval, decoder_final_state = \
        build_attention_model_np(params, dict_var_vals, src_vocab, trg_vocab,
                                 source_ids, source_seq_length,
                                 beam_size=beam_size,
                                 max_step=100)
    pred_ids = np.stack(decoder_output_eval['predicted_ids'])
    beam_ids = np.stack(decoder_output_eval['beam_ids'])
    log_probs = decoder_final_state.log_probs

    # beam decode
    gathered_pred_ids = np.zeros_like(beam_ids)
    for idx in range(beam_ids.shape[0]):
        gathered_pred_ids = gathered_pred_ids[:,
                            beam_ids[idx] % beam_ids.shape[1]]
        gathered_pred_ids[idx, :] = pred_ids[idx]

    seq_lens = []
    for idx in range(beam_ids.shape[1]):
        pred_ids_list = gathered_pred_ids[:, idx].tolist()
        seq_lens.append(pred_ids_list.index(
            trg_vocab.eos_id) + 1 if trg_vocab.eos_id in pred_ids_list else len(
            pred_ids_list))

    log_probs_np = log_probs / np.array(seq_lens)

    pids = gathered_pred_ids[:, np.argmax(log_probs_np)].tolist()

    print(trg_vocab.id_to_token(pids))


if __name__ == '__main__':
    configs = get_config('word2pos')

    sentence = sys.argv[1]
    print('Translating: {}'.format(sentence))
    infer(configs.src_vocab, configs.trg_vocab,
          sentence, configs.params, beam_size=5,
          model_dir='models/')
