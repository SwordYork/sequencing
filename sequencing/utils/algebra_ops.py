#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import tensorflow as tf

from .. import DTYPE


class LinearOp(object):
    def __init__(self, input_size, output_size, init_scale=None,
                 name='linear_op'):
        if not init_scale:
            init_scale = (6./(input_size + output_size)) ** 0.5

        with tf.variable_scope(name):
            self.W = tf.get_variable('weights',
                                     shape=[input_size, output_size],
                                     initializer=tf.random_uniform_initializer(
                                         -init_scale,
                                         init_scale),
                                     dtype=DTYPE)
            self.b = tf.get_variable('biases', shape=[output_size], dtype=DTYPE)

    def __call__(self, inputs):
        return tf.add(tf.matmul(inputs, self.W), self.b)


class LookUpOp(object):
    """Returns the embedding used for the sequence."""

    def __init__(self, vocab_size, embedding_dim, init_scale=0.01,
                 name='look_up_op'):
        with tf.variable_scope(name):
            self.table = tf.get_variable(
                name='W',
                shape=[vocab_size, embedding_dim],
                initializer=tf.random_uniform_initializer(-init_scale,
                                                          init_scale),
                dtype=DTYPE)

    def __call__(self, input_ids):
        return tf.nn.embedding_lookup(self.table, input_ids)
