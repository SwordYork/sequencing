#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from collections import namedtuple

from .decoder import Decoder
from .. import np, DTYPE, MODE
from ..nn import rnn_cells
from ..nn.algebra_ops import LinearOp


class AttentionRNNDecoder(Decoder):
    """
    An RNN Decoder that uses attention over an input sequence.
    """

    def __init__(self,
                 params,
                 attention,
                 feedback,
                 init_states,
                 mode=MODE.TRAIN,
                 name='attention_decoder'):
        super(AttentionRNNDecoder, self).__init__(params)

        num_layers = params['rnn_cell']['num_layers']
        cell_name = params['rnn_cell']['cell_name']
        self.num_layers = num_layers

        self.cells = []
        for i in range(num_layers):
            layer_name = 'decoder/multi_rnn_cell/cell_{}'.format(i)
            self.cells.append(getattr(rnn_cells, cell_name)(init_states[i],
                                                            base_name=layer_name))

        self.attention = attention
        self.feedback = feedback
        self.batch_size = self.feedback.batch_size

        self.init_states = [cell.init_state for cell in self.cells]

        self.attention_mix = LinearOp(base_name=name, name='attention_mix')
        self.logits_trans = LinearOp(base_name=name, name='logits_trans')

        self.mode = mode
        if self.mode != MODE.INFER:
            self.output_tuple = namedtuple('output',
                                           ['logits', 'predicted_ids'])
            self.state_tuple = namedtuple('decoder_state', ['cell_states'])
        else:
            self.output_tuple = namedtuple('output', ['logits', 'predicted_ids',
                                                      'beam_ids'])
            self.state_tuple = namedtuple('beam_decoder_state',
                                          ['cell_states', 'log_probs',
                                           'finished'])

    def initialize(self):
        finished, first_inputs = self.feedback.initialize()

        # Concat empty attention context
        attention_context = np.zeros([
            self.batch_size,
            self.attention.values.shape[-1]
        ], dtype=DTYPE)

        initial_state = self.init_states

        first_inputs = np.concatenate([first_inputs, attention_context], axis=1)

        if self.mode == MODE.INFER:
            log_probs = np.zeros([self.batch_size, ], dtype=DTYPE)
            return finished, first_inputs, self.state_tuple(initial_state,
                                                            log_probs, finished)

        return finished, first_inputs, self.state_tuple(initial_state)

    def finalize(self, final_outputs, final_state):
        return final_outputs, final_state

    def compute_output(self, cell_output):
        # Compute attention
        att_scores, attention_context = self.attention.compute_scores(
            query=cell_output)

        # TODO: verify whether it is necessary, or follow with a non-linear transform
        softmax_input = np.tanh(self.attention_mix(
            np.concatenate([cell_output, attention_context], 1)))

        # Softmax computation
        logits = self.logits_trans(softmax_input)

        return softmax_input, logits, att_scores, attention_context

    def step(self, time, inputs, state):
        if self.mode == MODE.INFER:
            return self._beam_step(time, inputs, state)

        return self._eval_step(time, inputs, state)

    @staticmethod
    def mask_finished(finished, now_, prev_):
        mask = np.expand_dims(finished, 1)

        if isinstance(prev_, tuple):
            # tuple states
            next_ = tuple((1. - mask) * ns + mask * s for ns, s in zip(
                now_, prev_))
        else:
            next_ = (1. - mask) * now_ + mask * prev_

        return next_

    def _eval_step(self, time, inputs, state):
        states = state.cell_states
        next_inputs = inputs
        cell_states = []
        for cl in range(self.num_layers):
            # forward
            next_inputs, cell_state = self.cells[cl].step(states[cl],
                                                          next_inputs)
            cell_states.append(cell_state)

        cell_output = next_inputs
        cell_output_new, logits, attention_scores, attention_context = \
            self.compute_output(cell_output)

        sample_ids = self.feedback.sample(logits=logits)

        outputs = self.output_tuple(
            logits=logits,
            predicted_ids=sample_ids)

        if self.mode == MODE.TRAIN:
            # just for test
            finished, next_inputs = self.feedback.next_inputs(time=time)
        else:
            finished, next_inputs = self.feedback.next_inputs(time=time,
                                                              sample_ids=sample_ids)

        next_inputs = np.concatenate([next_inputs, attention_context], 1)

        # once finished, always EOS
        next_state = self.mask_finished(finished, cell_states, states)
        next_inputs = self.mask_finished(finished, next_inputs, inputs)

        return (outputs, self.state_tuple(next_state), next_inputs, finished)

    def _beam_step(self, time, inputs, state):
        states = state.cell_states
        next_inputs = inputs
        cell_states = []
        for cl in range(self.num_layers):
            # forward
            next_inputs, cell_state = self.cells[cl].step(states[cl],
                                                          next_inputs)
            cell_states.append(cell_state)

        cell_output = next_inputs
        cell_output_new, logits, attention_scores, attention_context = \
            self.compute_output(cell_output)

        sample_ids, beam_ids, log_probs = \
            self.feedback.sample(logits=logits,
                                 log_probs=state.log_probs,
                                 prev_finished=state.finished,
                                 time=time)

        # attention_scores = tf.gather(attention_scores, beam_ids)
        attention_context = attention_context[beam_ids]

        def gather_state(state, idx):
            if isinstance(state, tuple):
                c = state[0][idx]
                h = state[1][idx]
                return (c, h)
            elif isinstance(state, tuple):
                return state[idx]

        next_cell_states = tuple(
            [gather_state(state, beam_ids) for state in cell_states])

        outputs = self.output_tuple(
            logits=logits,
            predicted_ids=sample_ids,
            beam_ids=beam_ids)

        finished, next_inputs = self.feedback.next_inputs(time=time,
                                                          sample_ids=sample_ids)

        next_inputs = np.concatenate([next_inputs, attention_context], 1)
        next_state = self.state_tuple(cell_states=next_cell_states,
                                      log_probs=log_probs,
                                      finished=finished)

        return (outputs, next_state, next_inputs, finished)
