#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import json

import numpy
import sequencing as sq
import tensorflow as tf
from sequencing import TIME_MAJOR, MODE, LinearOp
from sequencing.utils.metrics import Delta_BLEU
from sequencing.utils.misc import get_rnn_cell, EncoderDecoderBridge


def cross_entropy_sequence_loss(logits, targets, sequence_length):
    with tf.name_scope('cross_entropy_sequence_loss'):
        total_length = tf.to_float(tf.reduce_sum(sequence_length))
        batch_size = tf.to_float(tf.shape(sequence_length)[0])

        entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
        loss_mask = tf.transpose(tf.to_float(loss_mask), [1, 0])

        losses = entropy_losses * loss_mask
        # losses.shape: T * B
        # sequence_length: B
        total_loss_avg = tf.reduce_sum(losses) / batch_size

        return total_loss_avg


def rl_sequence_loss(logits, predict_ids, sequence_length,
                     baseline_states, reward, start_rl_step):
    # reward: T * B
    with tf.name_scope('rl_sequence_loss'):
        max_ml_step = tf.to_int32(tf.maximum(tf.reduce_max(start_rl_step), 0))
        min_ml_step = tf.to_int32(tf.maximum(tf.reduce_min(start_rl_step), 0))

        # entropy loss:
        # before start_rl_step is ml entropy
        # after start_rl_step should be rl entropy
        entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=predict_ids)

        # ML loss
        ml_entropy_losses = tf.slice(entropy_losses, [0, 0], [max_ml_step, -1])

        # Mask out the losses we don't care about
        ml_loss_mask = tf.sequence_mask(
            tf.to_int32(start_rl_step), max_ml_step)
        ml_loss_mask = tf.transpose(tf.to_float(ml_loss_mask), [1, 0])

        ml_loss = tf.reduce_sum(ml_entropy_losses * ml_loss_mask) / \
                  tf.maximum(tf.reduce_sum(ml_loss_mask), 1)

        # RL
        rl_entropy_losses = tf.slice(entropy_losses, [min_ml_step, 0], [-1, -1])

        # Mask out the losses we don't care about
        rl_loss_mask = (
            tf.to_float(tf.sequence_mask(
                tf.to_int32(sequence_length - min_ml_step),
                tf.to_int32(tf.shape(predict_ids)[0] - min_ml_step)))
            - tf.to_float(tf.sequence_mask(
                tf.to_int32(start_rl_step - min_ml_step),
                tf.to_int32(tf.shape(predict_ids)[0] - min_ml_step))))

        rl_loss_mask = tf.transpose(tf.to_float(rl_loss_mask), [1, 0])
        baseline_states = tf.slice(baseline_states, [min_ml_step, 0, 0],
                                   [-1, -1, -1])

        reward = tf.slice(reward, [min_ml_step, 0], [-1, -1])
        # prevent from dividing by zero
        rl_total = tf.maximum(tf.reduce_sum(rl_loss_mask), 1)

        with tf.variable_scope('baseline'):
            reward_predicted_m = tf.contrib.layers.fully_connected(
                baseline_states, baseline_states.get_shape().as_list()[-1],
                activation_fn=tf.nn.relu, scope='middle')
            # note, there is no negative reward, so we could use relu
            reward_predicted = tf.contrib.layers.fully_connected(
                reward_predicted_m, 1, activation_fn=None)

        reward_predicted = tf.squeeze(reward_predicted, axis=[2])

        reward_losses = tf.pow(reward_predicted - reward, 2)
        reward_loss_rmse = tf.sqrt(
            tf.reduce_sum(reward_losses * rl_loss_mask) / rl_total + 1e-12)

        reward_entropy_losses = (reward - tf.stop_gradient(reward_predicted)) \
                                * rl_entropy_losses * rl_loss_mask
        reward_entropy_loss = tf.reduce_sum(reward_entropy_losses) / rl_total

        predict_reward = tf.cond(tf.greater(tf.shape(reward_predicted)[0], 0),
                                 lambda: tf.reduce_mean(
                                     tf.slice(reward_predicted, [0, 0],
                                              [1, -1])),
                                 lambda: tf.to_float(0))

        # Calculate the average log perplexity in each batch
        total_loss_avg = ml_loss + reward_entropy_loss + reward_loss_rmse
        # the first reward predict is total reward
        return total_loss_avg, \
               ml_loss, \
               reward_loss_rmse, \
               predict_reward


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

def build_attention_model(params, src_vocab, trg_vocab,
                          source_placeholders, target_placeholders,
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

    decoder_params = params['decoder']
    # parameters
    source_ids = source_placeholders['src']
    source_seq_length = source_placeholders['src_len']
    source_sample_matrix = source_placeholders['src_sample_matrix']
    source_word_seq_length = source_placeholders['src_word_len']

    target_ids = target_placeholders['trg']
    target_seq_length = target_placeholders['trg_len']

    # Because source encoder is different to the target feedback,
    # we construct source_embedding_table manually
    source_char_embedding_table = sq.LookUpOp(src_vocab.vocab_size,
                                         src_vocab.embedding_dim,
                                         name='source')
    source_char_embedded = source_char_embedding_table(source_ids)

    # encode char to word
    char_encoder = sq.StackRNNEncoder(params['encoder'],
                                      params['encoder']['attention_key_size'],
                                      name='char_rnn',
                                      mode=mode)

    # char_encoder_outputs: T_c B F
    char_encoded_representation = char_encoder.encode(source_char_embedded, source_seq_length)
    char_encoder_outputs = char_encoded_representation.outputs
    #dynamical_batch_size = tf.shape(char_encoder_outputs)[1]
    #space_indices = tf.where(tf.equal(tf.transpose(source_ids), src_vocab.space_id))
    ##space_indices = tf.transpose(tf.gather_nd(tf.transpose(space_indices), [[1], [0]]))
    #space_indices = tf.concat(tf.split(space_indices, 2, axis=1)[::-1], axis=1)
    #space_indices = tf.transpose(tf.reshape(space_indices, [dynamical_batch_size, -1, 2]),
    #                             [1, 0, 2])
    ## T_w * B * F
    #source_embedded = tf.gather_nd(char_encoder_outputs, space_indices)

    # must be time major
    char_encoder_outputs = tf.transpose(char_encoder_outputs, perm=(1, 0, 2))
    sampled_word_embedded = tf.matmul(source_sample_matrix, char_encoder_outputs)
    source_embedded = tf.transpose(sampled_word_embedded, perm=(1, 0, 2))

    char_attention_keys = char_encoded_representation.attention_keys
    char_attention_values = char_encoded_representation.attention_values
    char_attention_length = char_encoded_representation.attention_length

    encoder = sq.StackBidirectionalRNNEncoder(params['encoder'],
                                              params['encoder']['attention_key_size'],
                                              name='stack_rnn',
                                              mode=mode)
    encoded_representation = encoder.encode(source_embedded, source_word_seq_length)
    attention_keys = encoded_representation.attention_keys
    attention_values = encoded_representation.attention_values
    attention_length = encoded_representation.attention_length
    encoder_final_states_bw = encoded_representation.final_state[-1][-1]

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
        infer_states_bw_shape = encoder_final_states_bw.get_shape().as_list()[-1]

        infer_char_key_size = char_attention_keys.get_shape().as_list()[-1]
        infer_char_value_size = char_attention_values.get_shape().as_list()[-1]

        encoder_final_states_bw = tf.reshape(tf.tile(encoder_final_states_bw, [1, beam_size]),
                                  [-1, infer_states_bw_shape])

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

            final_char_key_shape = [-1, dynamical_batch_size * beam_size,
                                    infer_char_key_size]
            final_char_value_shape = [-1, dynamical_batch_size * beam_size,
                                      infer_char_value_size]
            char_attention_keys = tf.reshape(
                (tf.tile(char_attention_keys, [1, 1, beam_size])), final_char_key_shape)
            char_attention_values = tf.reshape(
                (tf.tile(char_attention_values, [1, 1, beam_size])),
                final_char_value_shape)

        else:
            dynamical_batch_size = tf.shape(attention_keys)[0]
            final_key_shape = [dynamical_batch_size * beam_size, -1,
                               infer_key_size]
            final_value_shape = [dynamical_batch_size * beam_size, -1,
                                 infer_value_size]
            final_char_key_shape = [dynamical_batch_size * beam_size, -1,
                               infer_char_key_size]
            final_char_value_shape = [dynamical_batch_size * beam_size, -1,
                                 infer_char_value_size]

            attention_keys = tf.reshape(
                (tf.tile(attention_keys, [1, beam_size, 1])), final_key_shape)
            attention_values = tf.reshape(
                (tf.tile(attention_values, [1, beam_size, 1])),
                final_value_shape)

            char_attention_keys = tf.reshape(
                (tf.tile(char_attention_keys, [1, beam_size, 1])), final_char_key_shape)
            char_attention_values = tf.reshape(
                (tf.tile(char_attention_values, [1, beam_size, 1])),
                final_char_value_shape)


        attention_length = tf.reshape(
            tf.transpose(tf.tile([attention_length], [beam_size, 1])), [-1])
        char_attention_length = tf.reshape(
            tf.transpose(tf.tile([char_attention_length], [beam_size, 1])), [-1])

        feedback = sq.BeamFeedBack(trg_vocab, beam_size, dynamical_batch_size,
                                   max_step=max_step)

    encoder_decoder_bridge = EncoderDecoderBridge(encoder_final_states_bw.get_shape().as_list()[-1],
                                              decoder_params['rnn_cell'])
    decoder_state_size = decoder_params['rnn_cell']['state_size']
    # attention
    attention = sq.AvAttention(decoder_state_size,
                             attention_keys, attention_values, attention_length,
                             char_attention_keys, char_attention_values, char_attention_length)
    context_size = attention.context_size

    with tf.variable_scope('logits_func'):
        attention_mix = LinearOp(
                    context_size + feedback.embedding_dim + decoder_state_size,
                    decoder_state_size , name='attention_mix')
        attention_mix_middle = LinearOp(
                    decoder_state_size, decoder_state_size // 2,
                    name='attention_mix_middle')
        logits_trans = LinearOp(decoder_state_size // 2, feedback.vocab_size,
                                name='logits_trans')
        logits_func = lambda _softmax: logits_trans(
                                        tf.nn.relu(attention_mix_middle(
                                        tf.nn.relu(attention_mix(_softmax)))))

    # decoder
    decoder = sq.AttentionRNNDecoder(decoder_params, attention,
                                     feedback,
                                     logits_func=logits_func,
                                     init_state=encoder_decoder_bridge(encoder_final_states_bw),
                                     mode=mode)
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
        # Creates a variable to hold the global_step.
        global_step_tensor = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step')[0]
        rl_time_steps = tf.floordiv(tf.maximum(global_step_tensor -
                                               burn_in_step, 0),
                                    increment_step)
        start_rl_step = target_seq_length - rl_time_steps

        baseline_states = tf.stop_gradient(decoder_output.baseline_states)
        predict_ids = tf.stop_gradient(decoder_output.predicted_ids)

        # TODO: bug in tensorflow
        ground_or_predict_ids = tf.cond(tf.greater(rl_time_steps, 0),
                                        lambda: predict_ids,
                                        lambda: ground_truth_ids)

        reward, sequence_length = tf.py_func(
            func=_py_func,
            inp=[ground_or_predict_ids, ground_truth_ids, trg_vocab.eos_id],
            Tout=[tf.float32, tf.int32],
            name='reward')
        sequence_length.set_shape((None,))

        total_loss_avg, entropy_loss_avg, reward_loss_rmse, reward_predicted \
            = rl_sequence_loss(
            logits=decoder_output.logits,
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
