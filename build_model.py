#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import json

import numpy
import tensorflow as tf

import sequencing as sq
from sequencing import TIME_MAJOR, MODE
from sequencing.utils.metrics import Delta_BLEU


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
    with tf.name_scope('cross_entropy_sequence_loss'):
        total_length = tf.to_float(tf.reduce_sum(sequence_length))

        entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
        loss_mask = tf.transpose(tf.to_float(loss_mask), [1, 0])

        losses = entropy_losses * loss_mask
        # losses.shape: T * B
        # sequence_length: B
        total_loss_avg = tf.reduce_sum(losses) / total_length

        return total_loss_avg


def rl_sequence_loss(logits, targets, predict_ids, sequence_length,
                     baseline_states, reward, start_rl_step):
    # reward: T * B
    with tf.name_scope('rl_sequence_loss'):
        max_ml_step = tf.to_int32(tf.maximum(tf.reduce_max(start_rl_step), 0))
        # ML loss
        ml_entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.slice(logits, [0, 0, 0], [max_ml_step, -1, -1]),
            labels=tf.slice(targets, [0, 0], [max_ml_step, -1]))

        # Mask out the losses we don't care about
        ml_loss_mask = tf.sequence_mask(
            tf.to_int32(start_rl_step), max_ml_step)
        ml_loss_mask = tf.transpose(tf.to_float(ml_loss_mask), [1, 0])

        ml_loss = tf.reduce_sum(ml_entropy_losses * ml_loss_mask) / \
                  tf.maximum(tf.reduce_sum(ml_loss_mask), 1)

        rl_entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=predict_ids)

        # Mask out the losses we don't care about
        rl_loss_mask = (
            tf.to_float(tf.sequence_mask(tf.to_int32(sequence_length),
                                         tf.to_int32(tf.shape(predict_ids)[0])))
            - tf.to_float(tf.sequence_mask(tf.to_int32(start_rl_step),
                                           tf.to_int32(
                                               tf.shape(predict_ids)[0]))))

        rl_loss_mask = tf.transpose(tf.to_float(rl_loss_mask), [1, 0])

        # prevent from dividing by zero
        rl_total = tf.maximum(tf.reduce_sum(rl_loss_mask), 1)

        with tf.variable_scope('baseline'):
            reward_predicted_m = tf.contrib.layers.fully_connected(
                baseline_states, baseline_states.get_shape().as_list()[-1],
                activation_fn=tf.nn.relu, scope='middle')
            # note, there is no negative reward, so we could use relu
            reward_predicted = tf.contrib.layers.fully_connected(
                reward_predicted_m, 1, activation_fn=None)

        reward_predicted = tf.squeeze(reward_predicted)

        reward_losses = tf.pow(reward_predicted - reward, 2)
        reward_loss_rmse = tf.sqrt(
            tf.reduce_sum(reward_losses * rl_loss_mask) / rl_total + 1e-12)

        reward_entropy_losses = (reward - tf.stop_gradient(reward_predicted)) \
                                * rl_entropy_losses * rl_loss_mask
        reward_entropy_loss = tf.reduce_sum(reward_entropy_losses) / rl_total
        # Calculate the average log perplexity in each batch
        total_loss_avg = ml_loss + reward_entropy_loss + reward_loss_rmse
        # the first reward predict is total reward
        return total_loss_avg, \
               ml_loss, \
               reward_loss_rmse, \
               tf.reduce_mean(tf.slice(reward_predicted, [0, 0], [1, -1]))


def _py_func(predict_target_ids, ground_truth_ids, eos_id):
    n = 4  # 4-gram
    delta = True  # delta future reward
    batch_size = predict_target_ids.shape[1]
    length = numpy.zeros(batch_size, dtype=numpy.int32)
    reward = numpy.zeros_like(predict_target_ids, dtype=numpy.float32)

    for i in range(batch_size):
        p_id = predict_target_ids[:, i].tolist()
        p_len = p_id.index(eos_id) + 1 if eos_id in p_id else len(p_id)
        length[i] = p_len
        p_id = p_id[:p_len]

        t_id = ground_truth_ids[:, i].tolist()
        t_len = t_id.index(eos_id) + 1 if eos_id in t_id else len(t_id)
        t_id = t_id[:t_len]

        bleu_scores = Delta_BLEU(p_id, t_id, n)
        reward_i = bleu_scores[:, n - 1].copy()

        if delta:
            reward_i[1:] = reward_i[1:] - reward_i[:-1]
            reward[:p_len, i] = reward_i[::-1].cumsum()[::-1]
        else:
            reward[:p_len, i] = reward_i[-1]

    return reward, length


def build_attention_model(params, src_vocab, trg_vocab, source_ids,
                          source_seq_length, target_ids, target_seq_length,
                          beam_size=1, mode=MODE.TRAIN,
                          burn_in_step=100000, increment_step=10000,
                          teacher_rate=1.0, max_step=100):
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
    :param target_seq_length: placeholder
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
    if mode == MODE.RL:
        tf.logging.info('BUILDING RL TRAIN FEEDBACK......')
        dynamical_batch_size = tf.shape(attention_keys)[1]
        feedback = sq.RLTrainingFeedBack(target_ids, target_seq_length,
                                         trg_vocab, dynamical_batch_size,
                                         burn_in_step=burn_in_step,
                                         increment_step=increment_step,
                                         max_step=max_step)
    elif mode == MODE.TRAIN:
        tf.logging.info('BUILDING TRAIN FEEDBACK WITH {} TEACHER_RATE'
                        '......'.format(teacher_rate))
        feedback = sq.TrainingFeedBack(target_ids, target_seq_length,
                                       trg_vocab, teacher_rate,
                                       max_step=max_step)
    elif mode == MODE.EVAL:
        tf.logging.info('BUILDING EVAL FEEDBACK ......')
        feedback = sq.TrainingFeedBack(target_ids, target_seq_length,
                                       trg_vocab, 0.,
                                       max_step=max_step)
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
                                                            swap_memory=True,
                                                            scope='decoder')

    # not training
    if mode == MODE.EVAL or mode == MODE.INFER:
        return decoder_output, decoder_final_state

    # bos is added in feedback
    # so target_ids is predict_ids
    if not TIME_MAJOR:
        ground_truth_ids = tf.transpose(target_ids, [1, 0])
    else:
        ground_truth_ids = target_ids

    # construct the loss
    if mode == MODE.RL:
        baseline_states = tf.stop_gradient(decoder_output.baseline_states)
        predict_ids = tf.stop_gradient(decoder_output.predicted_ids)

        reward, sequence_length = tf.py_func(
            func=_py_func,
            inp=[predict_ids, ground_truth_ids, trg_vocab.eos_id],
            Tout=[tf.float32, tf.int32],
            name='reward')
        sequence_length.set_shape((None,))

        # Creates a variable to hold the global_step.
        global_step_tensor = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step')[0]
        rl_time_steps = tf.floordiv(tf.maximum(global_step_tensor -
                                               burn_in_step, 0),
                                    increment_step)
        start_rl_step = target_seq_length - rl_time_steps

        total_loss_avg, entropy_loss_avg, reward_loss_rmse, reward_predicted \
            = rl_sequence_loss(
            logits=decoder_output.logits,
            targets=ground_truth_ids,
            predict_ids=predict_ids,
            sequence_length=sequence_length,
            baseline_states=baseline_states,
            start_rl_step=start_rl_step,
            reward=reward)
        return decoder_output, total_loss_avg, entropy_loss_avg, \
               reward_loss_rmse, reward_predicted
    else:
        total_loss_avg = cross_entropy_sequence_loss(
            logits=decoder_output.logits,
            targets=ground_truth_ids,
            sequence_length=target_seq_length)
        return decoder_output, total_loss_avg, total_loss_avg, \
               tf.to_float(0.), tf.to_float(0.)
