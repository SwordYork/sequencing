#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from abc import ABCMeta, abstractmethod

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from .. import MODE, TIME_MAJOR


class Decoder(metaclass=ABCMeta):
    """
    Decoder abstract class.
    It can be CNN, RNN or self-attention model.
    """

    def __init__(self, params, mode=MODE.TRAIN, *args, **kwargs):
        self.params = params
        self.time_major = TIME_MAJOR
        self.mode = mode

    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def finalize(self, final_outputs, final_state):
        raise NotImplementedError

    @abstractmethod
    def step(self, time, inputs, state):
        """
        Decode step to generate next output and state

        :param time:
        :param inputs: previous output, usually generated by feedback.
        :param state: previous decoder state
        :return:
        """
        raise NotImplementedError


def dynamic_decode(decoder,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
    """
    Perform dynamic decoding with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    Args:
      decoder: A `Decoder` instance.
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state)`.
    """
    with variable_scope.variable_scope(scope, 'decoder') as var_scope:
        # Properly cache variable values inside the while_loop
        if var_scope.caching_device is None:
            var_scope.set_caching_device(lambda op: op.device)

        initial_finished, initial_inputs, initial_state = decoder.initialize()
        initial_time = constant_op.constant(0, dtype=dtypes.int32)

        def _shape(batch_size, from_shape):
            if not isinstance(from_shape, tensor_shape.TensorShape):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(
                    ops.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tensor_shape.TensorShape([batch_size]).concatenate(
                    from_shape)

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state,
                      unused_inputs,
                      finished):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished):
            """Internal while_loop body.
            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
              ```
            """
            # finish should determined by decoder
            # state should be copied when finished, so that outputs would
            # always be EOS.
            (next_outputs, next_state, next_inputs,
             next_finished) = decoder.step(time, inputs, state)

            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, next_outputs)
            return (time + 1, outputs_ta, next_state, next_inputs, next_finished)

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs,
                initial_finished
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]

        final_outputs = nest.map_structure(lambda ta: ta.stack(),
                                           final_outputs_ta)

        final_outputs, final_state = decoder.finalize(final_outputs,
                                                      final_state)

    return final_outputs, final_state
