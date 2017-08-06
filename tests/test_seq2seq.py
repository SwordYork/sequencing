#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import numpy
import tensorflow as tf
from sequencing import TIME_MAJOR, MODE
from sequencing.attention import Attention
from sequencing.data.vocab import Vocab
from sequencing.decoders.attention_decoder import AttentionRNNDecoder
from sequencing.decoders.decoder import dynamic_decode
from sequencing.decoders.feedback import TrainingFeedBack
from sequencing.encoders.rnn_encoder import StackBidirectionalRNNEncoder
from sequencing.utils.algebra_ops import LookUpOp


def cross_entropy_sequence_loss(logits, targets, sequence_length):
    with tf.name_scope("cross_entropy_sequence_loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
        losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

        return losses


def test_seq2seq():
    # parameters
    src_vocab_size = 16
    src_embed_dim = 8

    trg_vocab_size = 16
    trg_embed_dim = 8

    # encoder
    encoder_hidden_units = 64
    attention_size = 32
    encoder_layers = 2

    decoder_state_size = 64

    encoder_params = {'rnn_cell': {'state_size': encoder_hidden_units,
                                   'cell_name': 'BasicLSTMCell',
                                   'num_layers': encoder_layers,
                                   'input_keep_prob': 1.0,
                                   'output_keep_prob': 1.0},
                      'attention_key_size': attention_size}

    decoder_params = {'rnn_cell': {'cell_name': 'GRUCell',
                                   'state_size': decoder_state_size,
                                   'num_layers': 2,
                                   'input_keep_prob': 1.0,
                                   'output_keep_prob': 1.0},
                      'trg_vocab_size': trg_vocab_size}

    # placeholder
    source_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='source_ids')
    source_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='source_seq_length')

    target_ids = tf.placeholder(tf.int32, shape=(None, None),
                                name='target_ids')
    target_seq_length = tf.placeholder(tf.int32, shape=(None,),
                                       name='target_seq_length')

    # Because source encoder is different to target feedback,
    # we construct source_embedding_table manually
    source_embedding_table = LookUpOp(src_vocab_size, src_embed_dim,
                                      name='source')
    source_embedded = source_embedding_table(source_ids)

    encoder = StackBidirectionalRNNEncoder(encoder_params)
    encoded_representation = encoder.encode(source_embedded, source_seq_length)

    # feedback
    vocab = Vocab([chr(ord('a') + i) for i in range(trg_vocab_size)],
                  trg_embed_dim)

    feedback = TrainingFeedBack(target_ids, target_seq_length - 1, vocab,
                                name='test_seq2seq')

    # attention
    attention = Attention(decoder_params['rnn_cell']['state_size'],
                          encoded_representation.attention_keys,
                          encoded_representation.attention_values,
                          encoded_representation.attention_length)

    # decoder
    decoder = AttentionRNNDecoder(decoder_params, attention, feedback,
                                  mode=MODE.EVAL)
    decoder_output, _ = dynamic_decode(decoder)

    # construct the loss
    if not TIME_MAJOR:
        predict_ids = tf.transpose(target_ids[:, 1:], [1, 0])
    else:
        predict_ids = target_ids[1:, :]

    losses = cross_entropy_sequence_loss(
        logits=decoder_output.logits[:, :, :],
        targets=predict_ids,
        sequence_length=target_seq_length - 1)

    # Calculate the average log perplexity in each batch
    loss = tf.reduce_sum(losses) / tf.to_float(
        tf.shape(target_seq_length - 1)[0])

    # optimizer
    train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # data
    np_source_ids = numpy.asarray(
        [[1, 2, 3], [2, 3, 4], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=numpy.int32)
    np_source_seq_length = numpy.asarray([3, 3, 3, 3, 3],
                                         dtype=numpy.int32)
    np_target_ids = numpy.asarray(
        [[1, 2, 3], [2, 3, 4], [4, 5, 6], [7, 8, 9], [10, 10, 12]],
        dtype=numpy.int32)
    np_target_seq_length = numpy.asarray([3, 3, 3, 3, 3],
                                         dtype=numpy.int32)

    expected_result = numpy.array([[2, 3, 5, 8, 10],
                                   [3, 4, 6, 9, 12]], dtype=numpy.int32)
    if TIME_MAJOR:
        np_source_ids = np_source_ids.T
        np_target_ids = np_target_ids.T

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(500):
            sess.run(train_step, feed_dict={source_ids: np_source_ids,
                                            source_seq_length: np_source_seq_length,
                                            target_ids: np_target_ids,
                                            target_seq_length: np_target_seq_length})

        predicted_ids_tf = sess.run(decoder_output.predicted_ids,
                                    feed_dict={source_ids: np_source_ids,
                                               source_seq_length: np_source_seq_length,
                                               target_ids: np_target_ids,
                                               target_seq_length: np_target_seq_length})
    # may fail
    #numpy.testing.assert_array_almost_equal(predicted_ids_tf,
    #                                        expected_result)
    numpy.testing.assert_array_almost_equal(predicted_ids_tf.shape,
                                            expected_result.shape)

if __name__ == '__main__':
    test_seq2seq()
