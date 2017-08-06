#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from .rnn import RNN
from .. import sigmoid
from ... import np


class BasicLSTMCell(RNN):
    def __init__(self, init_state, activation=None, base_name=None,
                 name=None, forget_bias=1.0):
        self.forget_bias = forget_bias
        super(BasicLSTMCell, self).__init__(init_state, ['kernel', 'bias'],
                                            activation, base_name, name)

    def step(self, prev_states, input_, mask=None):
        c = prev_states[0]
        h = prev_states[1]

        input_h = np.concatenate((input_, h), axis=1)
        concat_matmul = np.matmul(input_h, self.params['kernel']) \
                        + self.params['bias']
        i, j, f, o = np.split(concat_matmul, 4, axis=1)
        c = c * sigmoid(f + self.forget_bias) + sigmoid(i) * self.activation(j)
        h = self.activation(c) * sigmoid(o)

        c = prev_states[0] * (1 - mask[:, None]) \
            + c * mask[:, None] if mask is not None else c
        h = prev_states[1] * (1 - mask[:, None]) \
            + h * mask[:, None] if mask is not None else h
        output = np.zeros_like(h) * (1 - mask[:, None]) \
                 + h * mask[:, None] if mask is not None else h

        return output, (c, h)
