#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import tensorflow as tf

from .. import TIME_MAJOR, DTYPE
from ..utils.algebra_ops import LinearOp


class Attention(object):
    FLOAT_MIN = -100000.0

    def __init__(self, query_size, keys, values, values_length,
                 name='attention'):
        self.attention_size = keys.get_shape().as_list()[-1]
        self.context_size = values.get_shape().as_list()[-1]

        self.keys = keys
        self.values = values
        self.values_length = values_length
        self.query_trans = LinearOp(query_size, self.attention_size, name=name)

        with tf.variable_scope(name):
            self.v_att = tf.get_variable('v_att', shape=[self.attention_size],
                                         dtype=DTYPE)
            self.b_att = tf.get_variable('b_att', shape=[self.attention_size],
                                         dtype=DTYPE)


        self.time_axis = 0 if TIME_MAJOR else 1

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(self.keys)[self.time_axis]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(self.values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=DTYPE)

        if TIME_MAJOR:
            scores_mask = tf.transpose(scores_mask)

        self.scores_mask = scores_mask

    def compute_scores(self, query):
        att_query = self.query_trans(query)

        energies = tf.reduce_sum(self.b_att + self.v_att * tf.tanh(
            self.keys + tf.expand_dims(att_query, self.time_axis)), [2])

        # TODO: we should mask energies before stabilize
        energies = energies * self.scores_mask + (
            (1.0 - self.scores_mask) * self.FLOAT_MIN)

        # Stabilize energies first and then exp
        energies = energies - tf.reduce_max(energies, axis=self.time_axis,
                                            keep_dims=True)
        unnormalized_scores = tf.exp(energies) * self.scores_mask

        # TODO: scores_mask can not all be zeros
        normalization = tf.reduce_sum(unnormalized_scores, axis=self.time_axis,
                                      keep_dims=True)

        # Normalize the scores
        scores_normalized = unnormalized_scores / normalization

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * self.values
        context = tf.reduce_sum(context, self.time_axis, name='context')
        context.set_shape([None, self.values.get_shape().as_list()[-1]])

        return scores_normalized, context
