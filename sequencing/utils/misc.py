#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import collections
from contextlib import ExitStack

from .algebra_ops import LinearOp

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple



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
        elif k.endswith('_keep_prob'):
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


def optimistic_restore(session, save_file, graph=None):
    """
    Only load matched variables. For example, Adam may not be saved and not
    necessary to load.

    :param session:
    :param save_file: file path of the checkpoint.
    :return:
    """
    with ExitStack() as stack:
        if graph is not None:
            stack.enter_context(graph.as_default())
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted(
            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
             if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(
            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()),
                tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)

            saver = tf.train.Saver(restore_vars)
            saver.restore(session, save_file)
        tf.logging.info('restored {} variables: {}'.format(len(restore_vars),
                                                           restore_vars))

class EncoderDecoderBridge(object):

    def __init__(self, state_size, decoder_rnn_params):
        self.drnn_state_is_tuple = decoder_rnn_params['cell_name'].endswith('LSTMCell')
        self.drnn_num_layers = decoder_rnn_params['num_layers']
        self.drnn_state_size = decoder_rnn_params['state_size']
        if self.drnn_state_is_tuple:
            self.init_transfer = LinearOp(state_size,
                                          self.drnn_state_size * 2,
                                          name='init_transfer')
        else:
            self.init_transfer = LinearOp(state_size,
                                          self.drnn_state_size,
                                          name='init_transfer')

    def __call__(self, states):
        state_transfered = self.init_transfer(states)
        if self.drnn_state_is_tuple:
            initial_state = LSTMStateTuple(h=state_transfered[:,:self.drnn_state_size],
                                           c=state_transfered[:,self.drnn_state_size:])
            return tuple(initial_state for _ in range(self.drnn_num_layers))

        return tuple(state_transfered for _ in range(self.num_layers))


