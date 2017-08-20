# see https://github.com/rizar/actor-critic-public/blob/master/lvsr/error_rate.py
import math
from collections import defaultdict

import numpy


def Delta_BLEU(candidate, reference, n=4, smooth=True):
    """

    :param candidate:  list of ids
    :param reference:  list of ids
    :return:
    """

    bleu_scores = numpy.zeros((len(candidate), n))

    # count reference ngrams
    ref_counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(reference) - k + 1):
            ref_counts[tuple(reference[i:i + k])] += 1

    # for each partial sequence, calculate bleu
    ref_len = len(reference)
    pred_counts = defaultdict(int)
    correct = numpy.zeros(4)
    for i in range(1, len(candidate) + 1):
        for k in range(i, max(-1, i - n), -1):
            ngram = tuple(candidate[k - 1:i])
            # UNK token is not considered here
            pred_counts[ngram] += 1
            if pred_counts[ngram] <= ref_counts.get(ngram, 0):
                correct[len(ngram) - 1] += 1

        # compute partial bleu score
        bleu = 1.
        for j in range(n):
            if smooth:
                possible = max(0, i - j)
                bleu *= float(correct[j] + 1.) / (possible + 1.)
            else:
                possible = max(1, i - j)
                bleu *= float(correct[j]) / (possible)

            bleu_scores[i - 1, j] = bleu ** (1. / (j + 1))

        # brevity penalty
        if i < ref_len:
            ratio = (i + 1e-15) / (ref_len + 1e-9)
            bleu_scores[i - 1, :] *= math.exp(1 - 1 / ratio)

    return bleu_scores.astype('float32')


if __name__ == "__main__":
    candidate = "Find the closest length of reference to that of candidate".split()
    references = "Finds the closest length of reference to that of candidates".split()
    bleu = Delta_BLEU(candidate, references, smooth=False)
    numpy.testing.assert_array_almost_equal(bleu[-1][-1], 0.759836)

    candidate = "Find the closest length of reference to that of candidate".split()
    references = "Find closest length of reference to that of candidates".split()
    bleu = Delta_BLEU(candidate, references, smooth=False)
    numpy.testing.assert_array_almost_equal(bleu[-1][-1], 0.6606328)

    candidate = "Find closest length of reference to that of candidate".split()
    references = "Finds the closest length of reference to that of candidates".split()
    bleu = Delta_BLEU(candidate, references, smooth=False)
    numpy.testing.assert_array_almost_equal(bleu[-1][-1], 0.6496350)
