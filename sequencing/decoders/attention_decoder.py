#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from .decoder import Decoder
from .. import MODE, DTYPE
from ..utils.algebra_ops import LinearOp
from ..utils.misc import get_rnn_cell, merge_dict


class AttentionRNNDecoder(Decoder):
    """
    An RNN Decoder that uses attention over an input sequence.
    """

    def __init__(self,
                 params,
                 attention,
                 feedback,
                 mode=MODE.TRAIN,
                 name='attention_decoder'):

        super(AttentionRNNDecoder, self).__init__(params, mode)
        self.params = merge_dict(self._default_params(), params)
        self.state_size = self.params['rnn_cell']['state_size']
        self.vocab_size = feedback.vocab_size

        self.attention = attention
        self.context_size = self.attention.values.get_shape()[-1]
        self.feedback = feedback
        self.cell = tf.nn.rnn_cell.MultiRNNCell(get_rnn_cell(self.params[
                                                                 'rnn_cell']))
        with tf.variable_scope(name):
            self.attention_mix = LinearOp(self.context_size + self.state_size,
                                          self.state_size, name='attention_mix')
            self.logits_trans = LinearOp(self.state_size, self.vocab_size,
                                         name='logits_trans')
        self.batch_size = self.feedback.batch_size

        if self.mode == MODE.INFER:
            self.state_tuple = namedtuple('beam_decoder_state',
                                          ['cell_states', 'log_probs',
                                           'finished'])
        else:
            self.state_tuple = namedtuple('decoder_state', ['cell_states'])

        if self.mode == MODE.TRAIN or self.mode == MODE.EVAL:
            self.output_tuple = namedtuple('output',
                                           ['logits', 'predicted_ids'])
            self.output_size = self.output_tuple(logits=self.vocab_size,
                                                 predicted_ids=tf.TensorShape(
                                                     []))
            self.output_dtype = self.output_tuple(logits=DTYPE,
                                                  predicted_ids=tf.int32)
        elif self.mode == MODE.RL:
            self.output_tuple = namedtuple('output',
                                           ['logits', 'baseline_states',
                                            'predicted_ids'])
            self.output_size = self.output_tuple(logits=self.vocab_size,
                                                 baseline_states=self.state_size,
                                                 predicted_ids=tf.TensorShape(
                                                     []))
            self.output_dtype = self.output_tuple(logits=DTYPE,
                                                  baseline_states=DTYPE,
                                                  predicted_ids=tf.int32)
        else:
            self.output_tuple = namedtuple('output', ['logits', 'predicted_ids',
                                                      'beam_ids'])
            self.output_size = self.output_tuple(logits=self.vocab_size,
                                                 predicted_ids=tf.TensorShape(
                                                     []),
                                                 beam_ids=tf.TensorShape([]))
            self.output_dtype = self.output_tuple(logits=DTYPE,
                                                  predicted_ids=tf.int32,
                                                  beam_ids=tf.int32)

    def _default_params(self):
        return {'rnn_cell': {'cell_name': 'GRUCell',
                             'state_size': 64,
                             'num_layers': 2,
                             'input_keep_prob': 1.0,
                             'output_keep_prob': 1.0},
                'logits': {'input_keep_prob': 1.0}}

    def initialize(self):
        finished, first_inputs = self.feedback.initialize()

        # Concat empty attention context
        attention_context = tf.zeros([
            self.batch_size,
            self.attention.values.get_shape().as_list()[-1]
        ])

        initial_state = self.cell.zero_state(self.batch_size, dtype=DTYPE)
        first_inputs = tf.concat([first_inputs, attention_context], 1)

        if self.mode != MODE.INFER:
            return finished, first_inputs, self.state_tuple(
                cell_states=initial_state)
        else:
            log_probs = tf.zeros([self.batch_size, ])
            state = self.state_tuple(cell_states=initial_state,
                                     log_probs=log_probs,
                                     finished=finished)
            return finished, first_inputs, state

    def finalize(self, final_outputs, final_state):
        return final_outputs, final_state

    def compute_output(self, cell_output):
        # Compute attention
        att_scores, attention_context = self.attention.compute_scores(
            query=cell_output)

        # TODO: verify whether it is necessary, or follow with a non-linear transform
        softmax_input = tf.nn.tanh(self.attention_mix(
            tf.concat([cell_output, attention_context], 1)))

        if self.params['logits']['input_keep_prob'] < 1.0:
            softmax_input = tf.layers.dropout(softmax_input,
                                              1. - self.params['logits'][
                                                  'input_keep_prob'])

        # Softmax computation
        logits = self.logits_trans(softmax_input)

        return softmax_input, logits, att_scores, attention_context

    def step(self, time, inputs, state):
        if self.mode != MODE.INFER:
            return self._train_step(time, inputs, state)

        return self._beam_step(time, inputs, state)

    @staticmethod
    def mask_finished(finished, now_, prev_):
        mask = tf.expand_dims(tf.to_float(finished), 1)

        if isinstance(prev_, tuple):
            # tuple states
            next_ = []
            for ns, s in zip(now_, prev_):
                # fucking LSTMStateTuple
                if isinstance(ns, LSTMStateTuple):
                    next_.append(
                        LSTMStateTuple(c=(1. - mask) * ns.c + mask * s.c,
                                       h=(1. - mask) * ns.h + mask * s.h))
                else:
                    next_.append((1. - mask) * ns + mask * s)
            next_ = tuple(next_)
        else:
            next_ = (1. - mask) * now_ + mask * prev_

        return next_

    def _train_step(self, time, inputs, state):
        cell_output, cell_states = self.cell(inputs, state.cell_states)
        cell_output_new, logits, attention_scores, attention_context = \
            self.compute_output(cell_output)

        sample_ids = self.feedback.sample(logits=logits, time=time)

        if self.mode == MODE.RL:
            outputs = self.output_tuple(
                logits=logits,
                baseline_states=cell_output,
                predicted_ids=sample_ids)
        else:
            outputs = self.output_tuple(
                logits=logits,
                predicted_ids=sample_ids)

        finished, next_inputs = self.feedback.next_inputs(time=time,
                                                          sample_ids=sample_ids)

        next_inputs = tf.concat([next_inputs, attention_context], 1)

        # We don't mask state and outputs in train step, it should be masked as:
        if self.mode == MODE.TRAIN or self.mode == MODE.RL:
            next_state = self.state_tuple(cell_states=cell_states)
        else:
            # once finished, always EOS
            next_state = self.mask_finished(finished, cell_states,
                                            state.cell_states)
            next_state = self.state_tuple(cell_states=next_state)
            next_inputs = self.mask_finished(finished, next_inputs, inputs)

        return (outputs, next_state, next_inputs, finished)

    def _beam_step(self, time, inputs, state):
        cell_output, cell_states = self.cell(inputs, state.cell_states)
        cell_output_new, logits, attention_scores, attention_context = \
            self.compute_output(cell_output)

        sample_ids, beam_ids, log_probs = \
            self.feedback.sample(logits=logits,
                                 log_probs=state.log_probs,
                                 prev_finished=state.finished,
                                 time=time)

        # attention_scores = tf.gather(attention_scores, beam_ids)
        attention_context = tf.gather(attention_context, beam_ids)

        def gather_state(state, idx):
            if isinstance(state, tf.Tensor):
                return tf.gather(state, idx)
            elif isinstance(state, tf.contrib.rnn.LSTMStateTuple):
                c = tf.gather(state.c, idx)
                h = tf.gather(state.h, idx)
                return tf.contrib.rnn.LSTMStateTuple(c, h)
            else:
                raise ValueError(
                    "Unrecognized state type: %s" % str(type(state)))

        next_cell_states = tuple(
            [gather_state(state, beam_ids) for state in cell_states])

        outputs = self.output_tuple(
            logits=logits,
            predicted_ids=sample_ids,
            beam_ids=beam_ids)

        finished, next_inputs = self.feedback.next_inputs(time=time,
                                                          sample_ids=sample_ids)

        next_inputs = tf.concat([next_inputs, attention_context], 1)
        next_state = self.state_tuple(cell_states=next_cell_states,
                                      log_probs=log_probs,
                                      finished=finished)

        return (outputs, next_state, next_inputs, finished)
