 # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib


from functools import reduce
from collections import defaultdict

import tensorflow as tf

FLAGS = None


def average_ckpt(checkpoint_from_paths,
                 checkpoint_to_path):
    """Migrates the names of variables within a checkpoint.
    Args:
      checkpoint_from_path: Path to source checkpoint to be read in.
      checkpoint_to_path: Path to checkpoint to be written out.
    """
    with ops.Graph().as_default():
        new_variable_map = defaultdict(list)
        for checkpoint_from_path in checkpoint_from_paths:
            logging.info('Reading checkpoint_from_path %s' % checkpoint_from_path)
            reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_from_path)
            name_shape_map = reader.get_variable_to_shape_map()
            for var_name in name_shape_map:
                tensor = reader.get_tensor(var_name)
                new_variable_map[var_name].append(tensor)

        variable_map = {}
        for var_name in name_shape_map:
            tensor = reduce(lambda x, y: x + y, new_variable_map[var_name]) / len(new_variable_map[var_name])
            var = variables.Variable(tensor, name=var_name)
            variable_map[var_name] = var
      
        print(variable_map)
        saver = saver_lib.Saver(variable_map)
      
        with session.Session() as sess:
          sess.run(variables.global_variables_initializer())
          logging.info('Writing checkpoint_to_path %s' % checkpoint_to_path)
          saver.save(sess, checkpoint_to_path)
    
    logging.info('Summary:')
    logging.info('  Converted %d variable name(s).' % len(new_variable_map))


def main():
    from_ckpts = ['./model.ckpt-722500', './model.ckpt-721500']
    to_ckpt = './model.ckpt-1000000'
    average_ckpt(from_ckpts, to_ckpt)

if __name__ == '__main__':
    main()


