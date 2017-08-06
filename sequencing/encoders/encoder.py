#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from abc import ABCMeta, abstractmethod
from collections import namedtuple

from .. import MODE, TIME_MAJOR

EncoderOutput = namedtuple(
    'EncoderOutput',
    'outputs final_state attention_values attention_keys attention_length')


class Encoder(metaclass=ABCMeta):
    """
    Base class for all encoder.
    We just need to implement encode method
    """

    def __init__(self, params, mode=MODE.TRAIN, *args, **kwargs):
        self.params = params
        self.time_major = TIME_MAJOR
        self.mode = mode

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Encodes the inputs
        :param args:
            inputs: when time_major=False, shape is [B, T, ...]
                    when time_major=True, shape is [T, B, ...]
            sequence_length:
            There may be multiple inputs.
        :param kwargs:
        :return:
            EncoderOutput
        """
        raise NotImplementedError