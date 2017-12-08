#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops

from .. import TIME_MAJOR
from ..utils.algebra_ops import LookUpOp


def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
        dtype=inp.dtype, size=array_ops.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)


def _transpose_batch_time(x):
    """Transpose the batch and time dimensions of a Tensor.

    Retains as much of the static shape information as possible.

    Args:
      x: A tensor of rank 2 or higher.

    Returns:
      x transposed along the first two dimensions.

    Raises:
      ValueError: if `x` is rank 1 or lower.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError(
            "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
            (x, x_static_shape))
    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(
        x, array_ops.concat(
            ([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(
        tensor_shape.TensorShape([
            x_static_shape[1].value, x_static_shape[0].value
        ]).concatenate(x_static_shape[2:]))
    return x_t


class FeedBack(metaclass=ABCMeta):
    def __init__(self, vocab, max_step=-1, name=None, *args, **kwargs):
        """We need to provide some feedback to the decoder."""
        self.max_step = tf.convert_to_tensor([max_step], name='max_step')
        self.name = name
        self.bos_id = vocab.bos_id
        self.vocab_size = vocab.vocab_size
        self.embedding_dim = vocab.embedding_dim
        self.eos_id = vocab.eos_id

    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args):
        raise NotImplementedError

    @abstractmethod
    def next_inputs(self, *args, **kwargs):
        raise NotImplementedError


class TrainingFeedBack(FeedBack):
    def __init__(self, input_ids, sequence_length, vocab, teacher_rate=1.,
                 max_step=300, name='feedback'):
        """
        FeedBack when training, i.e. teacher forcing feedback.

        :param input_ids: index of sequence, including end of sequence (EOS).
        :param sequence_length:
        :param vocab: object, see `data/vocab.py`
        :param teacher_rate: float, related to DAD, teacher forcing.
        :param name:
        """
        super(TrainingFeedBack, self).__init__(vocab, max_step, name=name)

        # We need to convert first, because we may input numpy array
        inputs = tf.convert_to_tensor(input_ids, name='inputs')

        self.teacher_rate = teacher_rate

        if TIME_MAJOR:
            self.batch_size = tf.shape(inputs)[1]  # should be dynamical
        else:
            self.batch_size = tf.shape(inputs)[0]  # should be dynamical
            inputs = _transpose_batch_time(inputs)

        if teacher_rate > 0.:
            self._input_tas = _unstack_ta(inputs)

        sequence_length = tf.convert_to_tensor(
            sequence_length, name='sequence_length')
        self.sequence_length = sequence_length

        with tf.variable_scope(name):
            self.lookup = LookUpOp(vocab.vocab_size, vocab.embedding_dim)

    def initialize(self):
        # finished means EOS is feed.
        # we should consider the loss after finishing.
        finished = tf.equal(0, self.sequence_length)
        inputs = self.lookup(tf.tile([self.bos_id], [self.batch_size]))

        return finished, inputs

    def sample(self, logits, time=None):
        sample_ids = tf.cast(tf.argmax(logits, axis=-1), dtypes.int32)
        return sample_ids

    def next_inputs(self, time, sample_ids=None, prev_finished=None):
        if sample_ids is None or self.teacher_rate > 0.:
            finished = tf.greater_equal(time + 1, self.sequence_length)
        else:
            finished = math_ops.logical_or(
                tf.greater_equal(time + 1, self.max_step),
                tf.equal(self.eos_id, sample_ids))

        if self.teacher_rate == 1. or (sample_ids is None):
            next_input_ids = self._input_tas.read(time)
            return finished, self.lookup(next_input_ids)

        if self.teacher_rate > 0.:
            # scheduled
            teacher_rates = tf.less_equal(
                tf.random_uniform(tf.shape(sample_ids), minval=0., maxval=1.),
                self.teacher_rate)
            teacher_rates = tf.to_int32(teacher_rates)

            next_input_ids = (teacher_rates * self._input_tas.read(time)
                              + (1 - teacher_rates) * sample_ids)
        else:
            next_input_ids = sample_ids

        return finished, self.lookup(next_input_ids)


class RLTrainingFeedBack(FeedBack):
    def __init__(self, input_ids, sequence_length, vocab, batch_size,
                 burn_in_step, increment_step, max_step=300,
                 name='feedback'):
        """
        FeedBack when using RL, i.e. greedy feedback. But incrementally greedy.

        :param input_ids: index of sequence, including end of sequence (EOS).
        :param sequence_length:
        :param vocab: object, see `data/vocab.py`
        :param batch_size: should be dynamical
        :param burn_in_step:
        :param increment_step:
        :param max_step:
        :param name:
        """
        super(RLTrainingFeedBack, self).__init__(vocab, max_step, name=name)

        # We need to convert first, because we may input numpy array
        inputs = tf.convert_to_tensor(input_ids, name='inputs')

        if not TIME_MAJOR:
            inputs = _transpose_batch_time(inputs)

        self._input_tas = _unstack_ta(inputs)

        sequence_length = tf.convert_to_tensor(
            sequence_length, name='sequence_length')
        self.sequence_length = sequence_length

        # Creates a variable to hold the global_step.
        self.global_step_tensor = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step')[0]

        self.burn_in_step = burn_in_step
        self.increment_step = increment_step
        self.batch_size = batch_size
        self.max_sequence_length = tf.shape(inputs)[0]
        with tf.variable_scope(name):
            self.lookup = LookUpOp(vocab.vocab_size, vocab.embedding_dim)

    def initialize(self):
        # finished means EOS is feed.
        finished = tf.equal(0, self.max_step)
        finished = tf.tile(finished, [self.batch_size])
        inputs = self.lookup(tf.tile([self.bos_id], [self.batch_size]))

        return finished, inputs

    def sample(self, logits, time):
        rl_time_steps = tf.floordiv(tf.maximum(self.global_step_tensor -
                                               self.burn_in_step, 0),
                                    self.increment_step)
        start_rl_step = self.sequence_length - rl_time_steps

        next_input_ids = tf.cond(
            tf.greater_equal(time, self.max_sequence_length),
            lambda: tf.tile([self.eos_id], [self.batch_size]),
            lambda: self._input_tas.read(time))

        next_predicted_ids = tf.squeeze(tf.multinomial(logits, 1), axis=[-1])
        mask = tf.to_int32(time >= start_rl_step)

        return (1 - mask) * tf.to_int32(next_input_ids) + mask * tf.to_int32(
            next_predicted_ids)

    def next_inputs(self, time, sample_ids, prev_finished):
        finished = math_ops.logical_or(
            tf.greater_equal(time + 1, tf.maximum(self.max_step,
                                                  self.max_sequence_length)),
            tf.equal(self.eos_id, sample_ids))
        next_finished = math_ops.logical_or(finished, prev_finished)
        return next_finished, self.lookup(sample_ids)


class BeamFeedBack(FeedBack):
    def __init__(self, vocab, beam_size, batch_size, max_step=300,
                 name='feedback'):
        super(BeamFeedBack, self).__init__(vocab, max_step, name=name)
        self._batch_size = batch_size  # num of source sentences
        self.batch_size = batch_size * beam_size  # keep compatibility
        self.beam_size = beam_size

        with tf.variable_scope(name):
            self.lookup = LookUpOp(vocab.vocab_size, vocab.embedding_dim)

    def initialize(self):
        # finished means EOS is feed.
        finished = tf.equal(0, self.max_step)
        finished = tf.tile(finished, [self.batch_size])
        inputs = self.lookup(tf.tile([self.bos_id], [self.batch_size]))

        return finished, inputs

    def sample(self, logits, log_probs, prev_finished, time):
        """
        sample based on logits.

        :param logits: [_batch_size * beam_size, vocab.vocab_size]
        :param log_probs: [_batch_size * beam_size,], log_probs of current
        decoded sequence.
        :param prev_finished: [_batch_size * beam_size,], indicate each beam
        is finished or not.
        :param time:
        :return:
        """

        # [_batch_size * beam_size, target_vocab_size]
        probs = tf.nn.log_softmax(logits)

        mask_tensor = [tf.float32.max] * self.vocab_size
        mask_tensor[self.eos_id] = -1.
        mask_tensor = tf.expand_dims(tf.constant(mask_tensor,
                                                 dtype=tf.float32), 0)
        mask_probs = (tf.expand_dims(tf.to_float(prev_finished), 1)
                      * mask_tensor + 1.) * probs

        # [_batch_size * beam_size, target_vocab_size]        
        log_probs = mask_probs + tf.expand_dims(log_probs, 1)
        log_probs = tf.reshape(tf.reshape(log_probs, [-1]),
                               [self._batch_size, -1])

        # flatten
        log_probs_flat = tf.cond(
            tf.convert_to_tensor(time) > 0, lambda: log_probs,
            lambda: tf.slice(log_probs, [0, 0], [-1, self.vocab_size]))

        next_log_probs, word_ids = tf.nn.top_k(log_probs_flat, k=self.beam_size)

        next_log_probs = tf.reshape(next_log_probs, [-1])
        word_ids = tf.reshape(word_ids, [-1])

        sample_ids = tf.mod(word_ids, self.vocab_size)

        # beam ids should be adjusted according to _batch_size
        beam_add = tf.tile([tf.range(self._batch_size)],
                           [self.beam_size, 1]) * self.beam_size

        beam_ids = tf.div(word_ids, self.vocab_size) \
                   + tf.reshape(tf.transpose(beam_add), [-1])

        return sample_ids, beam_ids, next_log_probs

    def next_inputs(self, time, sample_ids):
        finished = math_ops.logical_or(
            tf.greater_equal(time + 1, self.max_step),
            tf.equal(self.eos_id, sample_ids))
        return finished, self.lookup(sample_ids)
