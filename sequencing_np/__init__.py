#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from collections import namedtuple

try:
    import minpy.numpy as np
except Exception:
    import numpy as np

MODE = namedtuple('MODE', 'TRAIN EVAL INFER')(0, 1, 2)
TIME_MAJOR = True
DTYPE = np.float32

from .encoders.rnn_encoder import StackBidirectionalRNNEncoder
from .nn import sigmoid, relu, LinearOp, LookUpOp, log_softmax
from .nn.base import Graph
from .nn.rnn_cells import GRUCell, BasicLSTMCell, BasicRNNCell
from .decoders.feedback import TrainingFeedBackTest, BeamFeedBack
from .attention.attention import Attention
from .decoders.attention_decoder import AttentionRNNDecoder
from .decoders.decoder import decode_loop
from .data.vocab import Vocab

__all__ = ('GRUCell', 'BasicLSTMCell', 'BasicRNNCell',
           'sigmoid', 'relu', 'log_softmax', 'LinearOp', 'LookUpOp',
           'StackBidirectionalRNNEncoder', 'Graph', 'TrainingFeedBackTest',
           'BeamFeedBack', 'Attention', 'AttentionRNNDecoder', 'decode_loop',
           'Vocab')

__version__ = 0.1
