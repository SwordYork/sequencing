#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from .. import np, TIME_MAJOR, DTYPE
from ..nn.base import Layer


class Attention(Layer):
    FLOAT_MIN = -100000.0

    def __init__(self, keys, values, values_length, base_name=None,
                 name=None, **kwargs):
        self.keys = keys
        self.values = values
        self.values_length = values_length
        self.seq_mask = np.zeros(self.values.shape[:2], DTYPE)

        self.time_axis = 0 if TIME_MAJOR else 1
        for i, j in enumerate(self.values_length):
            if TIME_MAJOR:
                self.seq_mask[:j, i] = 1
            else:
                self.seq_mask[i, :j] = 1

        super(Attention, self).__init__(['weights', 'biases', 'v_att'],
                                        base_name, name, **kwargs)

    def compute_scores(self, query):
        att_query = np.matmul(query, self.params['weights']) \
                    + self.params['biases']
        energies = self.params['v_att'] * np.tanh(self.keys + np.expand_dims(
            att_query, self.time_axis))
        energies = np.sum(energies, 2)
        energies = energies * self.seq_mask + (
            (1.0 - self.seq_mask) * self.FLOAT_MIN)

        # Stabilize energies first and then exp
        energies = energies - np.max(energies, axis=self.time_axis,
                                     keepdims=True)
        unnormalized_scores = np.exp(energies) * self.seq_mask

        scores_normalized = unnormalized_scores / unnormalized_scores.sum(
            axis=self.time_axis, keepdims=True)

        context = np.expand_dims(scores_normalized, 2) * self.values
        context = np.sum(context, axis=self.time_axis)

        return scores_normalized, context
