#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from .encoder import Encoder, EncoderOutput
from .. import np
from ..nn import rnn_cells
from ..nn.algebra_ops import LinearOp


class StackBidirectionalRNNEncoder(Encoder):
    """
    Stacked bidirectional RNN encoder in numpy
    """

    def __init__(self, params, init_states, name=None, **kwargs):
        base_name = name or 'stack_bidirectional_rnn'
        self.cells_fw = []
        self.cells_bw = []

        num_layers = params['rnn_cell']['num_layers']
        cell_name = params['rnn_cell']['cell_name']

        for i in range(num_layers):
            layer_name = '{}/cell_{}/bidirectional_rnn/fw'.format(base_name, i)
            self.cells_fw.append(getattr(rnn_cells, cell_name)(init_states[i],
                                                               base_name=layer_name))
            layer_name = '{}/cell_{}/bidirectional_rnn/bw'.format(base_name, i)
            self.cells_bw.append(getattr(rnn_cells, cell_name)(init_states[i],
                                                               base_name=layer_name))
        self.num_layers = num_layers

        att_name = name or 'fully_connected'
        self.att_keys_linear_op = LinearOp(name=att_name)

    def encode(self, inputs, sequence_length=None):
        next_inputs = inputs
        states_fw = []
        states_bw = []

        for cl in range(self.num_layers):
            # forward
            next_input_fw, state_fw = self.cells_fw[cl].encode(next_inputs,
                                                               sequence_length)
            # backward
            next_input_bw, state_bw = self.cells_bw[cl].encode(next_inputs,
                                                               sequence_length,
                                                               reverse=True)

            next_inputs = np.concatenate((next_input_fw, next_input_bw), axis=2)
            states_fw.append(state_fw)
            states_bw.append(state_bw)

        attention_keys = self.att_keys_linear_op(next_inputs)

        return EncoderOutput(
            outputs=next_inputs,
            final_state=(tuple(states_fw), tuple(states_bw)),
            attention_values=next_inputs,
            attention_keys=attention_keys,
            attention_length=sequence_length)
