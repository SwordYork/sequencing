#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import random
from collections import deque

import numpy
import sequencing as sq
import tensorflow as tf
from sequencing import MODE, TIME_MAJOR


def build_vocab(vocab_file, embedding_dim, delimiter=' '):
    # construct vocab
    with open(vocab_file, 'r') as f:
        symbols = [s[:-1] for s in f.readlines()]
    vocab = sq.Vocab(symbols, embedding_dim, delimiter)
    return vocab


def build_parallel_inputs(src_vocab, trg_vocab, src_data_file,
                          trg_data_file, batch_size, buffer_size=16,
                          rand_append=True, mode=MODE.TRAIN):
    # data file should be preprocessed. For example, tokenize and remove long
    # lines.

    read_buffer = deque(maxlen=buffer_size)

    def _parallel_generator():
        should_stop_read = False
        with open(src_data_file) as src_data, open(trg_data_file) as trg_data:
            while True:
                if not read_buffer and should_stop_read:
                    # should_stop_read will never be True when TRAIN
                    break

                if not read_buffer:
                    # read_buffer is empty
                    # we read a lot of sentences to read_buffer for sorting and
                    # caching
                    buffer_batch = []
                    for _ in range(buffer_size * batch_size):
                        s = src_data.readline()
                        t = trg_data.readline()
                        if not s or not t:
                            if s:
                                tf.logging.warning(
                                    'The source data file contains '
                                    'more sentences!')
                            if t:
                                tf.logging.warning(
                                    'The target data file contains '
                                    'more sentences!')

                            if mode == MODE.TRAIN:
                                # one of the files is reaching end of file
                                tf.logging.info('Read from head ......')
                                src_data.seek(0)
                                trg_data.seek(0)
                                s = src_data.readline()
                                t = trg_data.readline()
                            else:
                                should_stop_read = True
                                break

                        # impossible for s, t to be None
                        buffer_batch.append((src_vocab.string_to_ids(s),
                                             trg_vocab.string_to_ids(t)))

                    # sort by length if train
                    if mode == MODE.TRAIN:
                        buffer_batch = sorted(buffer_batch,
                                              key=lambda l: len(l[1]))

                    total_batches = len(buffer_batch) // batch_size

                    # smaller batch
                    if len(buffer_batch) % batch_size > 0:
                        total_batches += 1

                    for i in range(total_batches):
                        if i == (total_batches - 1):
                            # take all in last
                            lines = buffer_batch[i * batch_size:]
                        else:
                            lines = buffer_batch[
                                    i * batch_size:(i + 1) * batch_size]

                        src_len_np = numpy.asarray([len(l[0]) for l in lines],
                                                   dtype=numpy.int32)
                        trg_len_np = numpy.asarray([len(l[1]) for l in lines],
                                                   dtype=numpy.int32)

                        if TIME_MAJOR:
                            # fill with eos
                            src_np = numpy.full((src_len_np.max(), len(lines)),
                                                src_vocab.eos_id,
                                                dtype=numpy.int32)
                            trg_np = numpy.full((trg_len_np.max(), len(lines)),
                                                trg_vocab.eos_id,
                                                dtype=numpy.int32)
                            for idx, l in enumerate(lines):
                                src_np[:len(l[0]), idx] = l[0]
                                trg_np[:len(l[1]), idx] = l[1]
                        else:
                            # fill with eos
                            src_np = numpy.full((len(lines), src_len_np.max()),
                                                src_vocab.eos_id,
                                                dtype=numpy.int32)
                            trg_np = numpy.full((len(lines), trg_len_np.max()),
                                                trg_vocab.eos_id,
                                                dtype=numpy.int32)
                            for idx, l in enumerate(lines):
                                src_np[idx, :len(l[0])] = l[0]
                                trg_np[idx, :len(l[1])] = l[1]

                        # shuffle batches
                        if mode == MODE.TRAIN and rand_append and \
                                random.randint(0, 1):
                            read_buffer.append((src_np, src_len_np,
                                                trg_np, trg_len_np))
                        else:
                            read_buffer.appendleft((src_np, src_len_np,
                                                    trg_np, trg_len_np))

                s, sl, t, tl = read_buffer.pop()
                yield s, sl, t, tl

    return _parallel_generator()
