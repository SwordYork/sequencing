#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import json

import sequencing as sq
import tensorflow as tf
from sequencing import TIME_MAJOR, MODE


def optimistic_restore(session, save_file):
    """
    Only load matched variables. For example, Adam may not be saved and not
    necessary to load.

    :param session:
    :param save_file: file path of the checkpoint.
    :return:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
         if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(
        zip(map(lambda x: x.name.split(':')[0], tf.global_variables()),
            tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def cross_entropy_sequence_loss(logits, targets, sequence_length):
    with tf.name_scope("cross_entropy_sequence_loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
        losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

        return losses


def build_attention_model(params, src_vocab, trg_vocab, source_ids,
                          source_seq_length, target_ids, target_seq_length,
                          beam_size=1, mode=MODE.TRAIN, teacher_rate=1.0,
                          max_step=100):
    """
    Build a model.

    :param params: dict.
     {encoder: {rnn_cell: {},
                ...},
      decoder: {rnn_cell: {},
                ...}}
      for example:
        {'encoder': {'rnn_cell': {'state_size': 512,
                                   'cell_name': 'BasicLSTMCell',
                                   'num_layers': 2,
                                   'input_keep_prob': 1.0,
                                   'output_keep_prob': 1.0},
                      'attention_key_size': attention_size},
        'decoder':  {'rnn_cell': {'cell_name': 'BasicLSTMCell',
                                   'state_size': 512,
                                   'num_layers': 1,
                                   'input_keep_prob': 1.0,
                                   'output_keep_prob': 1.0},
                      'trg_vocab_size': trg_vocab_size}}
    :param src_vocab: Vocab of source symbols.
    :param trg_vocab: Vocab of target symbols.
    :param source_ids: placeholder
    :param source_seq_length: placeholder
    :param target_ids: placeholder
    :param target_ids: placeholder
    :param beam_size: used in beam inference
    :param mode:
    :return:
    """
    if mode != MODE.TRAIN:
        params = sq.disable_dropout(params)

    tf.logging.info(json.dumps(params, indent=4))

    # parameters
    encoder_params = params['encoder']
    decoder_params = params['decoder']

    # Because source encoder is different to the target feedback,
    # we construct source_embedding_table manually
    source_embedding_table = sq.LookUpOp(src_vocab.vocab_size,
                                         src_vocab.embedding_dim,
                                         name='source')
    source_embedded = source_embedding_table(source_ids)

    encoder = sq.StackBidirectionalRNNEncoder(encoder_params, name='stack_rnn',
                                      mode=mode)
    encoded_representation = encoder.encode(source_embedded, source_seq_length)
    attention_keys = encoded_representation.attention_keys
    attention_values = encoded_representation.attention_values
    attention_length = encoded_representation.attention_length

    # feedback
    if mode == MODE.TRAIN:
        tf.logging.info('BUILDING TRAIN FEEDBACK WITH {} TEACHER_RATE'
                        '......'.format(teacher_rate))
        feedback = sq.TrainingFeedBack(target_ids, target_seq_length,
                                       trg_vocab, teacher_rate)
    elif mode == MODE.EVAL:
        tf.logging.info('BUILDING EVAL FEEDBACK ......')
        feedback = sq.TrainingFeedBack(target_ids, target_seq_length,
                                       trg_vocab, 0.)
    else:
        tf.logging.info('BUILDING INFER FEEDBACK WITH BEAM_SIZE {}'
                        '......'.format(beam_size))
        infer_key_size = attention_keys.get_shape().as_list()[-1]
        infer_value_size = attention_values.get_shape().as_list()[-1]

        # expand beam
        if TIME_MAJOR:
            # batch size should be dynamical
            dynamical_batch_size = tf.shape(attention_keys)[1]
            final_key_shape = [-1, dynamical_batch_size * beam_size,
                               infer_key_size]
            final_value_shape = [-1, dynamical_batch_size * beam_size,
                                 infer_value_size]
            attention_keys = tf.reshape(
                (tf.tile(attention_keys, [1, 1, beam_size])), final_key_shape)
            attention_values = tf.reshape(
                (tf.tile(attention_values, [1, 1, beam_size])),
                final_value_shape)
        else:
            dynamical_batch_size = tf.shape(attention_keys)[0]
            final_key_shape = [dynamical_batch_size * beam_size, -1,
                               infer_key_size]
            final_value_shape = [dynamical_batch_size * beam_size, -1,
                                 infer_value_size]
            attention_keys = tf.reshape(
                (tf.tile(attention_keys, [1, beam_size, 1])), final_key_shape)
            attention_values = tf.reshape(
                (tf.tile(attention_values, [1, beam_size, 1])),
                final_value_shape)

        attention_length = tf.reshape(
            tf.transpose(tf.tile([attention_length], [beam_size, 1])), [-1])

        feedback = sq.BeamFeedBack(trg_vocab, beam_size, dynamical_batch_size,
                                   max_step=max_step)

    # attention
    attention = sq.Attention(decoder_params['rnn_cell']['state_size'],
                             attention_keys, attention_values, attention_length)

    # decoder
    decoder = sq.AttentionRNNDecoder(decoder_params, attention,
                                     feedback, mode=mode)
    decoder_output, decoder_final_state = sq.dynamic_decode(decoder,
                                                            scope='decoder')

    if mode != MODE.TRAIN:
        return decoder_output, decoder_final_state

    # construct the loss
    # bos is added in feedback
    # so target_ids is predict_ids
    if not TIME_MAJOR:
        predict_ids = tf.transpose(target_ids, [1, 0])
    else:
        predict_ids = target_ids

    losses = cross_entropy_sequence_loss(
        logits=decoder_output.logits,
        targets=predict_ids,
        sequence_length=target_seq_length)

    return decoder_output, losses
