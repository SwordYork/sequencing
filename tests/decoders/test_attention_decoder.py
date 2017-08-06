#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import sequencing_np as snp
import tensorflow as tf
from sequencing import MODE
from sequencing.attention.attention import Attention
from sequencing.data.vocab import Vocab
from sequencing.decoders.attention_decoder import AttentionRNNDecoder
from sequencing.decoders.decoder import dynamic_decode
from sequencing.decoders.feedback import TrainingFeedBack
from sequencing_np import np, DTYPE
from sequencing_np.attention.attention import Attention as Attention_np
from sequencing_np.decoders.attention_decoder import \
    AttentionRNNDecoder as AttentionRNNDecoder_np
from sequencing_np.decoders.decoder import decode_loop
from sequencing_np.decoders.feedback import TrainingFeedBackTest


def test_attention_decoder():
    batch_size = 5
    attention_time_steps = 6
    attention_size = 7
    encoder_output_size = 9
    time_steps = 20
    vocab_size = 17
    embedding_dim = 12
    decoder_state_size = 19

    # ---------------------------
    # construct attention
    # random input
    if snp.TIME_MAJOR:
        attention_keys = np.asarray(
            np.random.rand(attention_time_steps, batch_size, attention_size),
            dtype=DTYPE)

        attention_values_np = np.asarray(
            np.random.rand(attention_time_steps, batch_size,
                           encoder_output_size), dtype=DTYPE)
    else:
        attention_keys = np.asarray(
            np.random.rand(batch_size, attention_time_steps, attention_size),
            dtype=DTYPE)

        attention_values_np = np.asarray(
            np.random.rand(batch_size, attention_time_steps,
                           encoder_output_size), dtype=DTYPE)

    attention_values = tf.convert_to_tensor(attention_values_np,
                                            name='attention_values')

    attention_values_length = np.random.randint(1, attention_time_steps + 1,
                                                batch_size)
    attention = Attention(decoder_state_size,
                          tf.convert_to_tensor(attention_keys),
                          attention_values, attention_values_length,
                          name='attention_decoder_test_decode')

    # ----------------------------------------------
    # construct feedback
    vocab = Vocab([chr(ord('a') + i) for i in range(vocab_size)],
                  embedding_dim)

    # dynamical batch size
    inputs = tf.placeholder(tf.int32, shape=(None, None),
                            name='source_ids')
    sequence_length = tf.placeholder(tf.int32, shape=(None,),
                                     name='source_seq_length')

    sequence_length_np = np.random.randint(1, time_steps + 1, batch_size,
                                           dtype=np.int32)

    # inputs to encoder
    if snp.TIME_MAJOR:
        inputs_np = np.random.randint(0, vocab_size, (time_steps, batch_size),
                                      dtype=np.int32)
    else:
        inputs_np = np.random.randint(0, vocab_size, (batch_size, time_steps),
                                      dtype=np.int32)

    feedback = TrainingFeedBack(inputs, sequence_length, vocab,
                                name='attention_decoder_test_decode')

    # --------------------------------------------------
    # construct decoder
    decoder_params = {'rnn_cell': {'cell_name': 'BasicLSTMCell',
                                   'state_size': decoder_state_size,
                                   'num_layers': 2,
                                   'input_keep_prob': 1.0,
                                   'output_keep_prob': 1.0},
                      'trg_vocab_size': vocab_size}

    decoder = AttentionRNNDecoder(decoder_params, attention, feedback,
                                  mode=MODE.EVAL,
                                  name='attention_decoder_test_decode')

    decoder_output, _ = dynamic_decode(decoder)

    init = tf.global_variables_initializer()
    # get weight
    train_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}
        decoder_output_tf = sess.run([decoder_output], feed_dict={inputs:
                                                                      inputs_np,
                                                                  sequence_length: sequence_length_np})

        init_state = decoder.cell.zero_state(batch_size, dtype=DTYPE)

        init_state_np = sess.run([init_state])[0]

    # test np implementation
    attention_np = Attention_np(attention_keys, attention_values_np,
                                attention_values_length,
                                name='attention_decoder_test_decode')

    feedback_np = TrainingFeedBackTest(inputs_np,
                                       sequence_length_np,
                                       vocab,
                                       name='attention_decoder_test_decode')

    decoder_np = AttentionRNNDecoder_np(decoder_params, attention_np,
                                        feedback_np, init_state_np,
                                        name='attention_decoder_test_decode')

    graph = snp.Graph()
    graph.initialize(dict_var_vals)

    decoder_output_np, _ = decode_loop(decoder_np)

    np.testing.assert_array_almost_equal(decoder_output_tf[0].logits,
                                         decoder_output_np['logits'])
    np.testing.assert_array_almost_equal(decoder_output_tf[0].logits,
                                         decoder_output_np['logits'])

    np.testing.assert_array_almost_equal(decoder_output_tf[0].predicted_ids,
                                         decoder_output_np['predicted_ids'])
    np.testing.assert_array_almost_equal(decoder_output_tf[0].predicted_ids,
                                         decoder_output_np['predicted_ids'])


if __name__ == '__main__':
    test_attention_decoder()
