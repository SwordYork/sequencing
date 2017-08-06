#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from .rnn import RNN
from ... import np


class BasicRNNCell(RNN):
    def __init__(self, init_state, activation=None, base_name=None, name=None):
        super(BasicRNNCell, self).__init__(init_state, ['kernel', 'bias'],
                                           activation, base_name, name)

    def step(self, prev_states, input_, mask=None):
        s = prev_states
        input_h = np.concatenate((input_, s), axis=1)
        h = self.activation(np.matmul(input_h, self.params['kernel']) +
                            self.params['bias'])
        h = s * (1 - mask[:, None]) \
            + h * mask[:, None] if mask is not None else h
        output = np.zeros_like(h) * (1 - mask[:, None]) \
                 + h * mask[:, None] if mask is not None else h
        return output, h
