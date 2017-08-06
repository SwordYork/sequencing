#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from .base import Layer
from .. import np


class LinearOp(Layer):
    def __init__(self, base_name=None, name=None, *args, **kwargs):
        super(LinearOp, self).__init__(['weights', 'biases'], base_name, name,
                                       **kwargs)

    def __call__(self, inputs):
        return np.matmul(inputs, self.params['weights']) + self.params['biases']


class LookUpOp(Layer):
    """Returns the embedding used for the sequence."""

    def __init__(self, base_name=None, name=None, *args, **kwargs):
        super(LookUpOp, self).__init__(['W'], base_name, name, **kwargs)

    def __call__(self, input_ids):
        return self.params['W'][input_ids]
