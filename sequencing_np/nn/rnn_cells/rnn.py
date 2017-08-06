#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from abc import ABCMeta, abstractmethod

from ..base import Layer
from ... import np, TIME_MAJOR


class RNN(Layer, metaclass=ABCMeta):
    def __init__(self, init_state, param_keys, activation=None,
                 base_name=None, name=None, *args, **kwargs):
        """
        numpy rnn cell.
        It only used for inferring, not training, thus we don't need initialization
        in this implementation.
        The weights and other things are passed by params.

        :param init_state: initial states of RNN, [B, H] or tuple([B, H], ...)
        :param param_keys: name of params, such as kernel and bias
        :param activation: activation function
        :param base_name: name of parent Layer
        :param name: name of this Layer
        """
        super(RNN, self).__init__(param_keys, base_name, name, **kwargs)
        # get state size
        if type(init_state) != type(np.empty([])):
            self.init_state = tuple(init_state)
            self.hidden_units = tuple(init_state)[0].shape[1]
        else:
            self.init_state = init_state
            self.hidden_units = init_state.shape[1]
        self.time_major = TIME_MAJOR
        self.activation = activation or np.tanh

    def encode(self, inputs, sequence_length=None, reverse=False):
        """
        Encode multi-step inputs.

        :param inputs: if time_major [T, B, ...] else [B, T, ...]
        :param sequence_length: length of the sequence [B]
        :param reverse: used in bidirectional RNN
        :return: lstm outputs
        """

        if not self.time_major:
            inputs = np.transpose(inputs, (1, 0, 2))

        steps = inputs.shape[0]
        outputs = np.zeros(inputs.shape[:-1] + (self.hidden_units,),
                           inputs.dtype)
        state = self.init_state

        iter_range = reversed(range(steps)) if reverse else range(steps)

        for idx in iter_range:
            # rnn step
            curr_input = inputs[idx, :, :]
            mask = idx < sequence_length if sequence_length is not None else None
            outputs[idx, :, :], state = self.step(state, curr_input, mask)

        if not self.time_major:
            outputs = np.transpose(outputs, (1, 0, 2))
        return outputs, state

    @abstractmethod
    def step(self, prev_states, input_, mask=None):
        """
        run rnn for one step
        :param prev_states: [B, ...]
        :param input_: [B, ...]
        :param mask: mask the terminated sequence in the batch
        :return: output, state
        """
        raise NotImplementedError
