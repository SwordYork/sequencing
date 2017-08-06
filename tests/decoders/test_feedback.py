#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
import sequencing_np as snp
import tensorflow as tf
from sequencing import TIME_MAJOR
from sequencing.data.vocab import Vocab
from sequencing.decoders.feedback import TrainingFeedBack
from sequencing_np import np
from sequencing_np.decoders.feedback import TrainingFeedBackTest


def test_training_feedback():
    batch_size = 2
    time_steps = 4
    vocab_size = 17
    embedding_dim = 12
    vocab = Vocab([chr(ord('a') + i) for i in range(vocab_size)], embedding_dim)

    # dynamical batch size
    inputs = tf.placeholder(tf.int32, shape=(None, None),
                            name='source_ids')
    sequence_length = tf.placeholder(tf.int32, shape=(None,),
                                     name='source_seq_length')

    feedback = TrainingFeedBack(inputs, sequence_length, vocab,
                                name='feedback_test')

    finished_list = []
    output_list = []

    for i in range(time_steps):
        outputs = feedback.next_inputs(i)
        finished_list.append(outputs[0])
        output_list.append(outputs[1])

    sequence_length_np = np.random.randint(1, time_steps + 1, batch_size,
                                           dtype=np.int32)
    sequence_length_np_b2 = np.random.randint(1, time_steps + 1,
                                              batch_size * 2,
                                              dtype=np.int32)
    if TIME_MAJOR:
        inputs_np = np.random.randint(0, vocab_size, (time_steps, batch_size),
                                      dtype=np.int32)

        inputs_np_b2 = np.random.randint(0, vocab_size,
                                         (time_steps, batch_size * 2),
                                         dtype=np.int32)
    else:
        inputs_np = np.random.randint(0, vocab_size, (batch_size, time_steps),
                                      dtype=np.int32)

        inputs_np_b2 = np.random.randint(0, vocab_size,
                                         (batch_size * 2, time_steps),
                                         dtype=np.int32)

    init = tf.global_variables_initializer()
    train_vars = tf.trainable_variables()
    with tf.Session() as sess:
        sess.run(init)
        train_vars_vals = sess.run(train_vars)
        dict_var_vals = {k.name.split(':')[0]: v for k, v in zip(train_vars,
                                                                 train_vars_vals)}

        tf_outputs = sess.run(finished_list + output_list,
                              feed_dict={inputs: inputs_np,
                                         sequence_length: sequence_length_np})

        tf_outputs_b2 = sess.run(finished_list + output_list,
                                 feed_dict={inputs: inputs_np_b2,
                                            sequence_length: sequence_length_np_b2})

    # print(inputs_np, tf_outputs, tf_outputs_b2, dict_var_vals)

    feedback_np = TrainingFeedBackTest(inputs_np, sequence_length_np, vocab,
                                       name='feedback_test')
    graph = snp.Graph()
    graph.initialize(dict_var_vals)

    for idx in range(time_steps):
        outputs_np = feedback_np.next_inputs(idx)
        np.testing.assert_array_almost_equal(outputs_np[1],
                                             tf_outputs[idx + time_steps])
        np.testing.assert_array_almost_equal(outputs_np[0],
                                             tf_outputs[idx])


if __name__ == '__main__':
    test_training_feedback()
