#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
from abc import ABCMeta, abstractmethod

from sequencing_np import np, DTYPE

from .. import TIME_MAJOR
from ..nn import log_softmax
from ..nn.algebra_ops import LookUpOp


class FeedBack(metaclass=ABCMeta):
    def __init__(self, vocab, max_step=-1, name=None, *args, **kwargs):
        """We need to provide some feedback to the decoder."""
        self.max_step = max_step

        self.name = name
        self.bos_id = vocab.bos_id
        self.vocab_size = vocab.vocab_size
        self.eos_id = vocab.eos_id

    @abstractmethod
    def initialize(self, *args):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args):
        raise NotImplementedError

    @abstractmethod
    def next_inputs(self, *args, **kwargs):
        raise NotImplementedError


# This is just for test, should not be used in np.
class TrainingFeedBackTest(FeedBack):
    def __init__(self, inputs, sequence_length, vocab, name=None):
        """
        FeedBack when training, i.e. teacher forcing feedback.

        :param input_ids: index of sequence, including end of sequence (EOS).
        :param sequence_length: should not including EOS, so we need to minus one.
        Because it at least including EOS, the min length is 0 not -1.
        :param vocab: object, see `data/vocab.py`
        """
        super(TrainingFeedBackTest, self).__init__(vocab=vocab)

        self.lookup = LookUpOp(base_name=name)
        self.sequence_length = sequence_length
        self.inputs = inputs

        if TIME_MAJOR:
            self.batch_size = self.inputs.shape[1]
        else:
            self.batch_size = self.inputs.shape[0]

    def initialize(self):
        # finished means EOS is feed.
        # we should consider the loss after finishing.
        finished = np.asarray(0 == self.sequence_length, dtype=DTYPE)
        inputs = self.lookup(np.tile(self.bos_id, self.batch_size))

        return finished, inputs

    def sample(self, logits):
        return np.argmax(logits, axis=1)

    def next_inputs(self, time):
        finished = (time + 1 >= self.sequence_length)
        if TIME_MAJOR:
            return finished, self.lookup(self.inputs[time, :])

        return finished, self.lookup(self.inputs[:, time])


# This is just for test, should not be used in np.
class BeamFeedBack(FeedBack):
    def __init__(self, vocab, beam_size=1, max_step=100, name=None):
        """
        FeedBack for beam decoding.

        :param vocab: object, see `data/vocab.py`
        :param beam_size:
        :param max_step: terminate if reach max_step
        :param name:
        """
        super(BeamFeedBack, self).__init__(vocab, max_step=max_step)

        self.lookup = LookUpOp(base_name=name)
        self.batch_size = beam_size

    def initialize(self):
        # finished means EOS is feed.
        finished = np.equal(0, self.max_step)
        finished = np.asarray(np.tile(finished, self.batch_size),
                              dtype=DTYPE)
        inputs = self.lookup(np.tile(self.bos_id, self.batch_size))

        return finished, inputs

    def sample(self, logits, log_probs, prev_finished, time):
        """
        sample based on logits.

        :param logits: [beam_size, vocab.vocab_size]
        :param log_probs: [beam_size,], log_probs of current decoded sequence.
        :param prev_finished: [beam_size,], indicate each beam is finished or not.
        :param time:
        :return:
        """
        probs = log_softmax(logits, axis=1)

        mask_tensor = [np.finfo(np.float32).max / np.max(-probs)] * \
                      self.vocab_size
        mask_tensor[self.eos_id] = -1.
        mask_tensor = np.asarray(mask_tensor, dtype=DTYPE)[None, :]

        # [beam_size, vocab_size]
        mask_probs = (prev_finished[:, None] * mask_tensor + 1.) * probs

        # [beam_size, vocab_size]
        log_probs = mask_probs + log_probs[:, None]

        log_probs_flat = log_probs[0, :] if time == 0 \
            else np.reshape(log_probs, [-1])

        word_ids = (-log_probs_flat).argsort()[:self.batch_size]
        next_log_probs = log_probs_flat[word_ids]

        sample_ids = np.mod(word_ids, self.vocab_size)
        beam_ids = np.floor_divide(word_ids, self.vocab_size, dtype=np.int32)
        return sample_ids, beam_ids, next_log_probs

    def next_inputs(self, time, sample_ids):
        finished = np.logical_or((time + 1 >= self.max_step),
                                 np.equal(self.eos_id, sample_ids))

        return finished, self.lookup(sample_ids)
