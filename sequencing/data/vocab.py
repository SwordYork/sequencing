#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
class Vocab(object):
    def __init__(self, tokens, embedding_dim, delimiter=' ', vocab_size=None,
                 bos_token='SEQ_BEG', eos_token='SEQ_END', unk_token='UNK'):
        """

        :param embedding_dim:
        :param tokens: list of tokens
        :param delimiter: delimiter between symbols, if '' then character level.
        :param vocab_size:
        :param bos_token:
        :param eos_token:
        :param unk_token:
        """
        if not vocab_size:
            vocab_size = len(tokens)
        vocab_size = min(vocab_size, len(tokens))

        self.embedding_dim = embedding_dim
        self.delimiter = delimiter

        self.unk_token, self.unk_id = unk_token, 0
        self.bos_token, self.bos_id = bos_token, 1
        self.eos_token, self.eos_id = eos_token, 2
        self.space_token, self.space_id = " ", 4
        self.token_to_id_dict = {token: token_id + 4 for token_id, token in
                                 enumerate(tokens[:vocab_size])}
        self.token_to_id_dict.update(**{self.unk_token: self.unk_id,
                                        self.bos_token: self.bos_id,
                                        self.eos_token: self.eos_id,
                                        self.space_token:self.space_id})

        self.id_to_token_dict = {v: k for k, v in self.token_to_id_dict.items()}
        self.vocab_size = vocab_size + 4
        #if self.delimiter == '':
        self.space_id = self.token_to_id_dict[' ']

    def _map_token_to_id_with_unk(self, token):
        try:
            return self.token_to_id_dict[token]
        except:
            return self.unk_id

    def _token_to_id(self, tokens):
        """

        :param tokens: tokens
        :return: list of ids
        """
        return list(map(self._map_token_to_id_with_unk, tokens))

    def string_to_ids(self, token_string, bos=False):
        if self.delimiter: # delimiter=" "
            tokens = token_string.strip().split(self.delimiter)
        else:
            # delimiter is '', character-level
            tokens = list(token_string.strip())
        if bos:
            return [self.bos_id] + self._token_to_id(tokens) + [self.eos_id]
        return self._token_to_id(tokens) + [self.eos_id]

    def id_to_token(self, token_ids, not_keep_eos=True):
        """

        :param token_ids: list of token ids
        :return:
        """
        token_len = token_ids.index(self.eos_id) \
            if (self.eos_id in token_ids) and not_keep_eos else len(token_ids)

        return (self.delimiter).join([self.id_to_token_dict[token_id] for
                                      token_id in token_ids[:token_len]])


def build_vocab(vocab_file, embedding_dim, delimiter=' ',
                vocab_size=None):
    # construct vocab
    with open(vocab_file, 'r') as f:
        symbols = [s.split('\t')[0] for s in f.readlines()]
    vocab = Vocab(symbols, embedding_dim, delimiter, vocab_size)
    return vocab
