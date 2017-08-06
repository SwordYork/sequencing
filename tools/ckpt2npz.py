#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import tensorflow as tf
import numpy

def save_as_npz(save_file):
    print(save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    numpy_dict = {}
    for k in saved_shapes:
        if not k.split('/')[-1].startswith('Adam'):
            numpy_dict[k] = reader.get_tensor(k)
    numpy.savez(save_file + '.npz', **numpy_dict)

last_ckpt = tf.train.latest_checkpoint('models')
save_as_npz(last_ckpt)