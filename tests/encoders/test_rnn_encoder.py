#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import sequencing_np as snp
import tensorflow as tf
from sequencing import MODE, TIME_MAJOR
from sequencing.encoders.rnn_encoder import StackBidirectionalRNNEncoder
from sequencing_np import np, DTYPE


def stack_bidir_rnn_encoder(rnn_cell, name=None):
    time_steps = 4
    hidden_units = 32
    batch_size = 6
    num_layers = 7
    input_size = 8
    attention_size = 9
    time_major = TIME_MAJOR

    params = {'rnn_cell': {'state_size': hidden_units,
                           'cell_name': rnn_cell,
                           'num_layers': num_layers,
                           'input_keep_prob': 1.0,
                           'output_keep_prob': 1.0},
              'attention_key_size': attention_size}

    encoder = StackBidirectionalRNNEncoder(params, mode=MODE.INFER, name=name)

    # inputs to encoder
    if time_major:
        inputs = np.asarray(np.random.rand(time_steps, batch_size, input_size),
                            dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)
    else:
        inputs = np.asarray(np.random.rand(batch_size, time_steps, input_size),
                            dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)
    output = encoder.encode(tf.convert_to_tensor(inputs), sequence_length)

    # get outputs of tensorflow
    init = tf.global_variables_initializer()
    train_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}
        output_tf = sess.run([output[0], output[1], output[2], output[3]])

    init_states = []
    for i in range(num_layers):
        if rnn_cell != 'BasicLSTMCell':
            init_states.append(np.zeros((batch_size, hidden_units),
                                        dtype=DTYPE))
            init_states.append(np.zeros((batch_size, hidden_units),
                                        dtype=DTYPE))
        else:
            init_states.append(
                (np.zeros((batch_size, hidden_units), dtype=DTYPE),) * 2)
            init_states.append(
                (np.zeros((batch_size, hidden_units), dtype=DTYPE),) * 2)

    encoder_np = snp.StackBidirectionalRNNEncoder(params, init_states, name)
    graph = snp.Graph()
    graph.initialize(dict_var_vals)

    output_np = encoder_np.encode(inputs, sequence_length)

    np.testing.assert_array_almost_equal(output_np[0], output_tf[0])
    np.testing.assert_array_almost_equal(output_np[1], output_tf[1])
    np.testing.assert_array_almost_equal(output_np[2], output_tf[2])
    np.testing.assert_array_almost_equal(output_np[3], output_tf[3])


def test_stack_bidir_rnn_encoder_rnn():
    graph = snp.Graph()
    graph.clear_layers()
    stack_bidir_rnn_encoder('BasicRNNCell', 'rnn')
    graph.clear_layers()


def test_stack_bidir_rnn_encoder_gru():
    graph = snp.Graph()
    graph.clear_layers()
    stack_bidir_rnn_encoder('GRUCell')
    graph.clear_layers()


def test_stack_bidir_rnn_encoder_lstm():
    graph = snp.Graph()
    graph.clear_layers()
    stack_bidir_rnn_encoder('BasicLSTMCell', 'bir_lstm')
    graph.clear_layers()


if __name__ == '__main__':
    test_stack_bidir_rnn_encoder_rnn()
    test_stack_bidir_rnn_encoder_gru()
    test_stack_bidir_rnn_encoder_lstm()
