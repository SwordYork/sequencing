#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from collections import namedtuple

import tensorflow

MODE = namedtuple('MODE', 'TRAIN EVAL INFER RL')(0, 1, 2, 3)
TIME_MAJOR = True
DTYPE = tensorflow.float32

from .encoders.rnn_encoder import StackBidirectionalRNNEncoder, StackRNNEncoder
from .attention.attention import Attention
from .attention.avattention import AvAttention
from .decoders.feedback import TrainingFeedBack, BeamFeedBack, \
    RLTrainingFeedBack
from .decoders.attention_decoder import AttentionRNNDecoder
from .decoders.decoder import dynamic_decode
from .utils.algebra_ops import LookUpOp, LinearOp
from .utils.misc import disable_dropout, optimistic_restore
from .data.vocab import Vocab, build_vocab

__all__ = ('StackBidirectionalRNNEncoder', 'StackRNNEncoder',
           'Attention', 'AvAttention', 'TrainingFeedBack',
           'RLTrainingFeedBack', 'BeamFeedBack', 'AttentionRNNDecoder',
           'dynamic_decode', 'LookUpOp', 'LinearOp', 'disable_dropout', 'Vocab',
           'optimistic_restore', 'build_vocab')


__version__ = 0.1
