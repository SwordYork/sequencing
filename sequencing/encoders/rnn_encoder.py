#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs

from .encoder import Encoder, EncoderOutput
from .. import MODE, DTYPE
from ..utils.misc import get_rnn_cell, merge_dict


# time_major is missing in tf 1.2, so I copied here.
def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    swap_memory=False,
                                    time_major=False,
                                    scope=None):
    """Creates a dynamic bidirectional recurrent neural network.
       Copy from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/rnn/python/ops/rnn.py
       But add swap_memory and time_major parameters.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to None.
    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs: Output `Tensor` shaped:
          `batch_size, max_time, layers_output]`. Where layers_output
          are depth-concatenated forward and backward outputs.
        output_states_fw is the final states, one tensor per layer,
          of the forward rnn.
        output_states_bw is the final states, one tensor per layer,
          of the backward rnn.
    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      ValueError: If inputs is `None`.
    """
    if not cells_fw:
        raise ValueError(
            "Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError(
            "Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError("Forward and Backward cells must have the same depth.")
    if initial_states_fw is not None and (not isinstance(cells_fw, list) or
                                                  len(cells_fw) != len(
                                                  cells_fw)):
        raise ValueError(
            "initial_states_fw must be a list of state tensors (one per layer).")
    if initial_states_bw is not None and (not isinstance(cells_bw, list) or
                                                  len(cells_bw) != len(
                                                  cells_bw)):
        raise ValueError(
            "initial_states_bw must be a list of state tensors (one per layer).")

    states_fw = []
    states_bw = []
    prev_layer = inputs

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                    time_major=time_major,
                    dtype=dtype)
                # Concat the outputs to create the new input.
                prev_layer = array_ops.concat(outputs, 2)
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


class StackBidirectionalRNNEncoder(Encoder):
    """Stacked Bidirectional RNN Encoder"""

    def __init__(self, params, mode=MODE.TRAIN, name=None):
        super(StackBidirectionalRNNEncoder, self).__init__(params, mode)
        self.name = name
        self.params = merge_dict(self._default_params(), params)

    def _default_params(self):
        return {'rnn_cell': {'cell_name': 'GRUCell',
                             'state_size': 64,
                             'num_layers': 2,
                             'input_keep_prob': 1.0,
                             'output_keep_prob': 1.0},
                'attention_key_size': 64}

    def encode(self, inputs, sequence_length, **kwargs):
        cells_fw = get_rnn_cell(self.params['rnn_cell'])
        cells_bw = get_rnn_cell(self.params['rnn_cell'])

        outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=DTYPE,
            scope=self.name,
            time_major=self.time_major,
            swap_memory=True,
            **kwargs)

        attention_keys = \
            tf.contrib.layers.fully_connected(outputs,
                                              self.params['attention_key_size'],
                                              activation_fn=None,
                                              scope=self.name)

        return EncoderOutput(
            outputs=outputs,
            final_state=(states_fw, states_bw),
            attention_values=outputs,
            attention_keys=attention_keys,
            attention_length=sequence_length)
