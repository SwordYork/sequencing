#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from sequencing_np.nn.base import Layer, Graph


class MyLayer(Layer):
    def __init__(self, base_name=None, name=None, dtype=None, **kwargs):
        super(MyLayer, self).__init__(['weight', 'bias'], base_name, name,
                                      dtype, **kwargs)


def test_base():
    graph = Graph()
    graph.clear_layers()
    my_layer = MyLayer()
    assert my_layer.name == 'my_layer'


def test_graph():
    MyLayer('graph')
    graph = Graph()
    assert len(graph.layers) == 2

    try:
        my_layer = MyLayer()
        graph.add_layers(my_layer)
    except ValueError:
        graph.clear_layers()
        return

    assert False


if __name__ == '__main__':
    test_base()
    test_graph()
