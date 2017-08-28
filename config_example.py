from collections import namedtuple

from build_inputs import build_vocab

TrainingConfigs = namedtuple('TrainingConfigs',
                             ['src_vocab', 'trg_vocab', 'params',
                              'train_src_file', 'train_trg_file',
                              'test_src_file', 'test_trg_file',
                              'beam_size', 'batch_size',
                              'max_step', 'model_dir',
                              'lr_rate', 'clip_gradient_norm',
                              'burn_in_step', 'increment_step', 'train_steps'])


def config_word2pos():
    # load vocab
    src_vocab = build_vocab('data/vocab.word', 256, ' ')
    trg_vocab = build_vocab('data/vocab.tag', 32, ' ')

    params = {'encoder': {'rnn_cell': {'state_size': 512,
                                       'cell_name': 'BasicLSTMCell',
                                       'num_layers': 1,
                                       'input_keep_prob': 1.0,
                                       'output_keep_prob': 1.0},
                          'attention_key_size': 256},
              'decoder': {'rnn_cell': {'cell_name': 'BasicLSTMCell',
                                       'state_size': 512,
                                       'num_layers': 1,
                                       'input_keep_prob': 1.0,
                                       'output_keep_prob': 1.0},
                          'logits': {'input_keep_prob': 1.0}}}

    configs = TrainingConfigs(
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        params=params,
        train_src_file='data/train.word',
        train_trg_file='data/train.tag',
        test_src_file='data/test.word',
        test_trg_file='data/test.tag',
        beam_size=1,
        batch_size=64,
        max_step=100,
        model_dir='models',
        lr_rate=0.001,
        clip_gradient_norm=5.,
        burn_in_step=3000,
        increment_step=1000,
        train_steps=200000)

    return configs


def config_en2zh():
    # load vocab
    src_vocab = build_vocab('data/vocab.en', 512, ' ')
    trg_vocab = build_vocab('data/vocab.zh', 512, '')

    params = {'encoder': {'rnn_cell': {'state_size': 1024,
                                       'cell_name': 'BasicLSTMCell',
                                       'num_layers': 1,
                                       'input_keep_prob': 1.0,
                                       'output_keep_prob': 1.0},
                          'attention_key_size': 512},
              'decoder': {'rnn_cell': {'cell_name': 'BasicLSTMCell',
                                       'state_size': 1024,
                                       'num_layers': 1,
                                       'input_keep_prob': 1.0,
                                       'output_keep_prob': 1.0},
                          'logits': {'input_keep_prob': 1.0}}}

    configs = TrainingConfigs(
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        params=params,
        train_src_file='data/en.tok.shuf.filter',
        train_trg_file='data/zh.tok.shuf.filter',
        test_src_file='data/test.en',
        test_trg_file='data/test.zh',
        beam_size=5,
        batch_size=128,
        max_step=150,
        model_dir='models',
        lr_rate=0.0005,
        clip_gradient_norm=5.,
        burn_in_step=50000,
        increment_step=10000,
        train_steps=200000)

    return configs


def config_en2zh_large():
    # load vocab
    src_vocab = build_vocab('data/vocab.en', 512, ' ')
    trg_vocab = build_vocab('data/vocab.zh', 512, '')

    params = {'encoder': {'rnn_cell': {'state_size': 1024,
                                       'cell_name': 'BasicLSTMCell',
                                       'num_layers': 4,
                                       'input_keep_prob': 0.95,
                                       'output_keep_prob': 1.0},
                          'attention_key_size': 512},
              'decoder': {'rnn_cell': {'cell_name': 'BasicLSTMCell',
                                       'state_size': 1024,
                                       'num_layers': 2,
                                       'input_keep_prob': 0.95,
                                       'output_keep_prob': 1.0},
                          'logits': {'input_keep_prob': 1.0}}}

    configs = TrainingConfigs(
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        params=params,
        train_src_file='data/en.tok.shuf.filter',
        train_trg_file='data/zh.tok.shuf.filter',
        test_src_file='data/test.en',
        test_trg_file='data/test.zh',
        beam_size=5,
        batch_size=128,
        max_step=150,
        model_dir='models',
        lr_rate=5e-4,
        clip_gradient_norm=5.,
        burn_in_step=100000,
        increment_step=20000,
        train_steps=400000)

    return configs
