from collections import namedtuple

from build_inputs import build_vocab

TrainingConfigs = namedtuple('TrainingConfigs',
                             ['src_vocab', 'trg_vocab', 'params',
                              'train_src_file', 'train_trg_file',
                              'test_src_file', 'test_trg_file',
                              'beam_size', 'batch_size',
                              'max_step', 'model_dir',
                              'lr_rate', 'rl_lr_rate',
                              'train_steps'])


def get_config(config_name):
    configs = [word2pos.__name__, en2zh.__name__]

    if config_name in configs:
        return eval(config_name)()
    else:
        raise Exception('Config not found')


def word2pos():
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
        rl_lr_rate=0.0001,
        train_steps=200000)

    return configs


def en2zh():
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
        rl_lr_rate=0.0001,
        train_steps=200000)

    return configs
