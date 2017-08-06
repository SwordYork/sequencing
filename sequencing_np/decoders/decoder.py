#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from abc import ABCMeta, abstractmethod

from .. import TIME_MAJOR, np


class Decoder(metaclass=ABCMeta):
    """
    Decoder abstract class.
    It can be CNN, RNN or self-attention model.
    """

    def __init__(self, params, *args, **kwargs):
        self.params = params
        self.time_major = TIME_MAJOR

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


def decode_loop(decoder):
    finished, inputs, state = decoder.initialize()
    time = 0
    outputs = {k: [] for k in decoder.output_tuple._fields}

    while not np.all(finished):
        (next_outputs, state, inputs, finished) = decoder.step(time, inputs,
                                                               state)
        next_outputs_dict = next_outputs._asdict()
        for k, v in next_outputs_dict.items():
            outputs[k].append(v)
        time += 1

    final_outputs, final_state = decoder.finalize(outputs, state)

    return final_outputs, final_state
