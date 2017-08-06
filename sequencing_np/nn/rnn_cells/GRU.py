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


class GRUCell(RNN):
    def __init__(self, init_state, activation=None, base_name=None, name=None):
        super(GRUCell, self).__init__(init_state,
                                      ['gates/kernel', 'gates/bias',
                                       'candidate/kernel', 'candidate/bias'],
                                      activation, base_name, name)

    def step(self, prev_states, input_, mask=None):
        s = prev_states
        input_s = np.concatenate((input_, s), axis=1)
        concat_matmul = sigmoid(np.matmul(input_s, self.params['gates/kernel'])
                                + self.params['gates/bias'])
        r, u = np.split(concat_matmul, 2, axis=1)

        input_h = np.concatenate((input_, r * s), axis=1)
        c = self.activation(np.matmul(input_h, self.params['candidate/kernel'])
                            + self.params['candidate/bias'])
        h = u * s + (1 - u) * c
        h = s * (1 - mask[:, None]) \
            + h * mask[:, None] if mask is not None else h
        output = np.zeros_like(h) * (1 - mask[:, None]) \
                 + h * mask[:, None] if mask is not None else h

        return output, h
