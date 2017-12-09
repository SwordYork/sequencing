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


class AvAttention(object):
    FLOAT_MIN = -100000.0

    def __init__(self, query_size, 
                 keys, values, values_length,
                 char_keys, char_values, char_values_length,
                 name='avattention'):
        self.attention_size = keys.get_shape().as_list()[-1]
        self.context_size = values.get_shape().as_list()[-1] \
                            + char_values.get_shape().as_list()[-1]

        self.keys = keys
        self.values = values
        self.values_length = values_length

        self.char_attention_size = char_keys.get_shape().as_list()[-1]
        self.char_keys = char_keys
        self.char_values = char_values
        self.char_values_length = char_values_length

        self.query_trans = LinearOp(query_size, 
                                    self.attention_size + self.char_attention_size,
                                    name=name)

        with tf.variable_scope(name):
            self.v_att = tf.get_variable('v_att', shape=[self.attention_size],
                                         dtype=DTYPE)
            self.b_att = tf.get_variable('b_att', shape=[self.attention_size],
                                         dtype=DTYPE)

            self.char_v_att = tf.get_variable('char_v_att', shape=[self.char_attention_size],
                                         dtype=DTYPE)
            self.char_b_att = tf.get_variable('char_b_att', shape=[self.char_attention_size],
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

        # Replace all char scores for padded inputs with tf.float32.min
        char_num_scores = tf.shape(self.char_keys)[self.time_axis]
        char_scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(self.char_values_length),
            maxlen=tf.to_int32(char_num_scores),
            dtype=DTYPE)

        if TIME_MAJOR:
            char_scores_mask = tf.transpose(char_scores_mask)

        self.char_scores_mask = char_scores_mask


    def compute_scores(self, query):
        full_att_query = self.query_trans(query)
        word_att_query = full_att_query[:, :self.attention_size]
        char_att_query = full_att_query[:, self.attention_size:]
        
        def _compute_context(v_att, b_att, att_query, keys, values, scores_mask):
            energies = tf.reduce_sum(b_att + v_att * tf.tanh(
                keys + tf.expand_dims(att_query, self.time_axis)), [2])
    
            # TODO: we should mask energies before stabilize
            energies = energies * scores_mask + (
                (1.0 - scores_mask) * self.FLOAT_MIN)
    
            # Stabilize energies first and then exp
            energies = energies - tf.reduce_max(energies, axis=self.time_axis,
                                                keep_dims=True)
            unnormalized_scores = tf.exp(energies) * scores_mask
    
            # TODO: scores_mask can not all be zeros
            normalization = tf.reduce_sum(unnormalized_scores, axis=self.time_axis,
                                          keep_dims=True)
    
            # Normalize the scores
            scores_normalized = unnormalized_scores / normalization
    
            # Calculate the weighted average of the attention inputs
            # according to the scores
            context = tf.expand_dims(scores_normalized, 2) * values
            context = tf.reduce_sum(context, self.time_axis, name='context')
            context.set_shape([None, values.get_shape().as_list()[-1]])
            
            return scores_normalized, context

        word_scores_normalized, word_context = _compute_context(self.v_att, self.b_att,
                                                      word_att_query, self.keys,
                                                      self.values, self.scores_mask)
                                                
        char_scores_normalized, char_context = _compute_context(self.char_v_att, self.char_b_att,
                                                      char_att_query, self.char_keys,
                                                      self.char_values, self.char_scores_mask)

        context = tf.concat([word_context, char_context], axis=1)

        return (word_scores_normalized, char_scores_normalized), context
