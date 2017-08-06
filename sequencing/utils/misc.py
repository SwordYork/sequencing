#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import collections

import tensorflow as tf


def merge_dict(d, u):
    """
    update d using u. It is used to merge model params.

    :param d: old dict
    :param u: new dict
    :return:
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = merge_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def disable_dropout(d):
    for k, v in d.items():
        if isinstance(v, collections.Mapping):
            r = disable_dropout(v)
            d[k] = r
        elif k == 'input_keep_prob' or k == 'output_keep_prob':
            d[k] = 1.0
    return d


def get_rnn_cell(params):
    cell_name = params['cell_name']
    state_size = params['state_size']
    num_layers = params['num_layers']
    input_keep_prob = params['input_keep_prob']
    output_keep_prob = params['output_keep_prob']

    cells = []
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.DropoutWrapper(
            getattr(tf.contrib.rnn, cell_name)(state_size),
            input_keep_prob, output_keep_prob)
        cells.append(cell)
    return cells
