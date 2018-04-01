#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import random
from collections import deque, namedtuple
import subprocess

import numpy
import tensorflow as tf

import sequencing as sq
from sequencing import MODE, TIME_MAJOR

def build_vocab(vocab_file, embedding_dim, delimiter=' '):
    # construct vocab
    with open(vocab_file, 'r') as f:
        symbols = [s[:-1] for s in f.readlines()]
    vocab = sq.Vocab(symbols, embedding_dim, delimiter)
    return vocab

def build_source_char_inputs(src_vocab, src_data_file,
                               batch_size, buffer_size=16,
                               mode=MODE.TRAIN):
    # data file should be preprocessed. For example, tokenize and remove long
    # lines.

    input_tuple = namedtuple('inputs',
                             ['src', 'src_len',
                              'src_sample_matrix','src_word_len'])

    def _source_generator():
        read_buffer = deque(maxlen=buffer_size)
        should_stop_read = False


        with open(src_data_file, 'r') as src_data:
            while True:
                if not read_buffer and should_stop_read:
                    break

                if not read_buffer:
                    # read_buffer is empty
                    # we read a lot of sentences to read_buffer for sorting and
                    # caching
                    buffer_batch = []
                    for _ in range(buffer_size * batch_size):
                        s = src_data.readline()
                        if not s:
                            should_stop_read = True
                            break

                        # impossible for s, t to be None
                        src_char_ids = src_vocab.string_to_ids(s)[:-1] \
                                       + [src_vocab.space_id, src_vocab.eos_id, src_vocab.space_id]

                        buffer_batch.append((src_char_ids,))

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

                        num_lines = len(lines)
                        src_word_len_np = numpy.asarray([l[0].count(src_vocab.space_id) for l in lines],
                                                     dtype=numpy.int32)
                        max_word_length = src_word_len_np.max()

                        src_len_np = numpy.asarray([len(l[0]) + max_word_length - src_word_len_np[li]
                                                    for li, l in enumerate(lines)],
                                                    dtype=numpy.int32)

                        src_sample_matrix_np = numpy.zeros((num_lines, max_word_length,
                                                            src_len_np.max()),
                                                          dtype=numpy.float32)

                        if TIME_MAJOR:
                            # fill with eos
                            src_np = numpy.full((src_len_np.max(), num_lines),
                                                src_vocab.eos_id,
                                                dtype=numpy.int32)
                            for idx, l in enumerate(lines):
                                src_np[:len(l[0]), idx] = l[0]
                                src_np[len(l[0]):src_len_np[idx], idx] = src_vocab.space_id
                                src_sample_matrix_np[idx, range(max_word_length),
                                        numpy.where(src_np[:, idx] == src_vocab.space_id)[0]] = 1.
                        else:
                            # fill with eos
                            src_np = numpy.full((num_lines, src_len_np.max()),
                                                src_vocab.eos_id,
                                                dtype=numpy.int32)
                            for idx, l in enumerate(lines):
                                src_np[idx, :len(l[0])] = l[0]
                                src_np[idx, len(l[0]):src_len_np[idx]] = src_vocab.space_id
                                src_sample_matrix_np[idx, range(max_word_length),
                                        numpy.where(src_np[:, idx] == src_vocab.space_id)[0]] = 1.

                        current_input_np = input_tuple(
                                src=src_np,
                                src_len=src_len_np,
                                src_sample_matrix=src_sample_matrix_np,
                                src_word_len=src_word_len_np)
                        read_buffer.appendleft(current_input_np)

                yield read_buffer.pop()

    return _source_generator()


def build_parallel_char_inputs(src_vocab, trg_vocab, src_data_file,
                               trg_data_file, batch_size, buffer_size=16,
                               rand_append=True, mode=MODE.TRAIN):
    # data file should be preprocessed. For example, tokenize and remove long
    # lines.

    input_tuple = namedtuple('inputs',
                             ['src', 'src_len',
                              'src_sample_matrix','src_word_len',
                              'trg', 'trg_len'])

    def _parallel_generator():
        read_buffer = deque(maxlen=buffer_size)
        should_stop_read = False

        if mode == MODE.TRAIN:
            tf.logging.info('Shuffling ......')
            subprocess.call(['./shuffle_data.sh', src_data_file, trg_data_file])
            src_data = open(src_data_file + '.shuf', 'r')
            trg_data = open(trg_data_file + '.shuf', 'r')
            tf.logging.info('Shuffle done ......')
        else:
            src_data = open(src_data_file, 'r')
            trg_data = open(trg_data_file, 'r')


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
                            src_data.close()
                            trg_data.close()

                            # shuf and reopen
                            tf.logging.info('Shuffling ......')
                            subprocess.call(['./shuffle_data.sh', src_data_file, trg_data_file])
                            src_data = open(src_data_file + '.shuf', 'r')
                            trg_data = open(trg_data_file + '.shuf', 'r')
                            tf.logging.info('Shuffle done ......')

                            s = src_data.readline()
                            t = trg_data.readline()
                        else:
                            src_data.close()
                            trg_data.close()
                            should_stop_read = True
                            break

                    # impossible for s, t to be None
                    src_char_ids = src_vocab.string_to_ids(s)[:-1] \
                                   + [src_vocab.space_id, src_vocab.eos_id, src_vocab.space_id]

                    buffer_batch.append((src_char_ids,
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

                    num_lines = len(lines)
                    src_word_len_np = numpy.asarray([l[0].count(src_vocab.space_id) for l in lines],
                                                 dtype=numpy.int32)
                    max_word_length = src_word_len_np.max()

                    src_len_np = numpy.asarray([len(l[0]) + max_word_length - src_word_len_np[li]
                                                for li, l in enumerate(lines)],
                                                dtype=numpy.int32)

                    trg_len_np = numpy.asarray([len(l[1]) for l in lines],
                                               dtype=numpy.int32)

                    src_sample_matrix_np = numpy.zeros((num_lines, max_word_length,
                                                        src_len_np.max()),
                                                      dtype=numpy.float32)

                    if TIME_MAJOR:
                        # fill with eos
                        src_np = numpy.full((src_len_np.max(), num_lines),
                                            src_vocab.eos_id,
                                            dtype=numpy.int32)
                        trg_np = numpy.full((trg_len_np.max(), num_lines),
                                            trg_vocab.eos_id,
                                            dtype=numpy.int32)
                        for idx, l in enumerate(lines):
                            src_np[:len(l[0]), idx] = l[0]
                            src_np[len(l[0]):src_len_np[idx], idx] = src_vocab.space_id
                            src_sample_matrix_np[idx, range(max_word_length),
                                    numpy.where(src_np[:, idx] == src_vocab.space_id)[0]] = 1.

                            trg_np[:len(l[1]), idx] = l[1]
                    else:
                        # fill with eos
                        src_np = numpy.full((num_lines, src_len_np.max()),
                                            src_vocab.eos_id,
                                            dtype=numpy.int32)
                        trg_np = numpy.full((num_lines, trg_len_np.max()),
                                            trg_vocab.eos_id,
                                            dtype=numpy.int32)
                        for idx, l in enumerate(lines):
                            src_np[idx, :len(l[0])] = l[0]
                            src_np[idx, len(l[0]):src_len_np[idx]] = src_vocab.space_id
                            src_sample_matrix_np[idx, range(max_word_length),
                                    numpy.where(src_np[:, idx] == src_vocab.space_id)[0]] = 1.

                            trg_np[idx, :len(l[1])] = l[1]

                    current_input_np = input_tuple(
                            src=src_np,
                            src_len=src_len_np,
                            src_sample_matrix=src_sample_matrix_np,
                            src_word_len=src_word_len_np,
                            trg=trg_np,
                            trg_len=trg_len_np)
                    read_buffer.appendleft(current_input_np)

                # shuffle batches
                if (mode == MODE.TRAIN or mode == MODE.RL) and rand_append:
                    random.shuffle(read_buffer)

            yield read_buffer.pop()

    return _parallel_generator()
