#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import sequencing_np as snp
import tensorflow as tf
from sequencing_np import np, DTYPE


def test_basic_rnn():
    hidden_units = 33
    batch_size = 13
    time_steps = 7
    embedding_size = 8
    time_major = snp.TIME_MAJOR

    # tensorflow results
    rnn = tf.contrib.rnn.BasicRNNCell(hidden_units, activation=tf.nn.relu)
    # initial state of the RNN
    state = np.repeat(
        np.asarray(np.random.rand(1, hidden_units), dtype=DTYPE),
        batch_size, axis=0)

    # inputs to RNN
    if time_major:
        inputs = np.asarray(np.random.rand(time_steps, batch_size,
                                           embedding_size), dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)
    else:
        inputs = np.asarray(np.random.rand(batch_size, time_steps,
                                           embedding_size), dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)

    outputs, _ = tf.nn.dynamic_rnn(cell=rnn,
                                   inputs=tf.convert_to_tensor(inputs),
                                   sequence_length=sequence_length,
                                   initial_state=state, dtype=DTYPE,
                                   time_major=time_major)

    # get outputs of tensorflow
    init = tf.global_variables_initializer()
    train_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}
        outputs_tf = sess.run(outputs)

    # test numpy implementation
    rnn_np = snp.BasicRNNCell(state, activation=snp.relu, base_name='rnn')
    graph = snp.Graph()
    graph.initialize(dict_var_vals)
    outputs_np, _ = rnn_np.encode(inputs, sequence_length=sequence_length)
    np.testing.assert_array_almost_equal(outputs_tf, outputs_np)


def test_gru():
    hidden_units = 33
    batch_size = 13
    time_steps = 7
    embedding_size = 8
    time_major = snp.TIME_MAJOR

    # tensorflow results
    gru = tf.contrib.rnn.GRUCell(hidden_units, activation=tf.nn.tanh)
    # initial state of the GRU
    state = np.repeat(
        np.asarray(np.random.rand(1, hidden_units), dtype=DTYPE),
        batch_size, axis=0)

    # inputs to GRU
    if time_major:
        inputs = np.asarray(np.random.rand(time_steps, batch_size,
                                           embedding_size), dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)
    else:
        inputs = np.asarray(np.random.rand(batch_size, time_steps,
                                           embedding_size), dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)

    outputs, _ = tf.nn.dynamic_rnn(cell=gru,
                                   inputs=tf.convert_to_tensor(inputs),
                                   sequence_length=sequence_length,
                                   initial_state=state, dtype=DTYPE,
                                   time_major=time_major)

    # get outputs of tensorflow
    init = tf.global_variables_initializer()
    train_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}
        outputs_tf = sess.run(outputs)

    # test numpy implementation
    gru_np = snp.GRUCell(state, activation=np.tanh, base_name='rnn')
    graph = snp.Graph()
    graph.initialize(dict_var_vals)
    outputs_np, _ = gru_np.encode(inputs, sequence_length=sequence_length)
    np.testing.assert_array_almost_equal(outputs_tf, outputs_np)


def test_lstm():
    hidden_units = 33
    batch_size = 13
    time_steps = 7
    embedding_size = 8
    forget_bias = 0.2
    time_major = snp.TIME_MAJOR

    # tensorflow results
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_units, activation=tf.nn.relu,
                                        forget_bias=forget_bias)
    # initial state of the LSTM
    state_c = np.repeat(
        np.asarray(np.random.rand(1, hidden_units), dtype=DTYPE),
        batch_size, axis=0)
    state_h = np.repeat(
        np.asarray(np.random.rand(1, hidden_units), dtype=DTYPE),
        batch_size, axis=0)
    state = (state_c, state_h)

    # inputs to LSTM
    if time_major:
        inputs = np.asarray(np.random.rand(time_steps, batch_size,
                                           embedding_size), dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)
    else:
        inputs = np.asarray(np.random.rand(batch_size, time_steps,
                                           embedding_size), dtype=DTYPE)
        sequence_length = np.random.randint(1, time_steps + 1, batch_size)

    outputs, _ = tf.nn.dynamic_rnn(cell=lstm,
                                   inputs=tf.convert_to_tensor(inputs),
                                   sequence_length=sequence_length,
                                   initial_state=tf.contrib.rnn.LSTMStateTuple(
                                       *state), dtype=DTYPE,
                                   time_major=time_major)

    # get outputs of tensorflow
    init = tf.global_variables_initializer()
    train_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}
        outputs_tf = sess.run(outputs)

    # test numpy implementation

    lstm_np = snp.BasicLSTMCell(state, activation=snp.relu,
                                forget_bias=forget_bias, base_name='rnn')
    graph = snp.Graph()
    graph.initialize(dict_var_vals)
    outputs_np, _ = lstm_np.encode(inputs, sequence_length=sequence_length)
    np.testing.assert_array_almost_equal(outputs_tf, outputs_np)


if __name__ == "__main__":
    test_basic_rnn()
    test_gru()
    test_lstm()
