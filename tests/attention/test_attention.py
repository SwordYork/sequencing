#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import sequencing_np as snp
import tensorflow as tf
from sequencing.attention.attention import Attention
from sequencing_np import np, DTYPE
from sequencing_np.attention.attention import Attention as Attention_np


def test_attention():
    batch_size = 5
    attention_time_steps = 6
    attention_size = 7
    state_size = 8
    encoder_output_size = 9

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

    attention = Attention(state_size, tf.convert_to_tensor(attention_keys),
                          attention_values, attention_values_length,
                          name='attention_test')

    # random query
    query = np.asarray(np.random.rand(batch_size, state_size),
                       dtype=DTYPE)

    # test normalized scores and context
    scores_normalized, context = attention.compute_scores(query)
    init = tf.global_variables_initializer()

    # get weight
    train_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}
        scores_normalized_tf, context_tf = sess.run(
            [scores_normalized, context])

    # test np implementation
    attention_np = Attention_np(attention_keys, attention_values_np,
                                attention_values_length,
                                name='attention_test')
    graph = snp.Graph()
    graph.initialize(dict_var_vals)

    scores_normalized_np, context_np = attention_np.compute_scores(query)

    np.testing.assert_array_almost_equal(scores_normalized_np,
                                         scores_normalized_tf)
    np.testing.assert_array_almost_equal(context_np, context_tf)


if __name__ == '__main__':
    test_attention()
