#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import re

from .. import DTYPE


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class Graph(metaclass=Singleton):
    """
    Graph should be a singleton.
    """
    def __init__(self):
        self.layers = {}

    def add_layers(self, layer):
        # collect layers
        if layer.name in self.layers:
            raise ValueError('Duplicated layer: {}'.format(layer.name))
        self.layers[layer.name] = layer

    def clear_layers(self):
        self.layers = {}

    def print_vars(self):
        # for debug
        for k in self.layers:
            print(self.layers[k].params)

    def initialize(self, trained_params):
        """
        Use tensorflow trained params to initialize np model

        :param trained_params: dict
        :return:
        """
        for k in self.layers:
            layer = self.layers[k]
            layer.initialize(trained_params)


class Layer(object):
    """Base layer class."""

    def __init__(self, params_keys, base_name=None, name=None,
                 dtype=DTYPE, **kwargs):
        """
        Base class of all layers.

        :param params_keys: name of params
        :param base_name: similar to scope
        :param name:  Layer name
        :param dtype:
        :param kwargs:
        """

        if not name:
            name = self.to_snake_case(self.__class__.__name__)

        # to match the trained params from tensorflow
        self.name = '{}/{}'.format(base_name, name) if base_name else name
        self.dtype = dtype

        self.params = dict.fromkeys(params_keys)
        graph = Graph()
        graph.add_layers(self)

    def initialize(self, trained_params):
        """
        Initialize this layer by the params from tensorflow.
        TODO: Try to match the convention in tensorflow

        :param trained_params: dict of trained params
        :return:
        """

        for k in self.params:
            key = '{}/{}'.format(self.name, k)
            self.params[k] = trained_params[key]

    @staticmethod
    def to_snake_case(name):
        intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
        insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
        # If the class is private the name starts with "_" which is not secure
        # for creating scopes. We prefix the name with "private" in this case.
        if insecure[0] != '_':
            return insecure
        return 'private' + insecure
