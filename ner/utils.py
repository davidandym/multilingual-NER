""" Simple util helpers. """


# pylint: disable=invalid-name


import numpy as np


PADKEY = '~@@PAD@@~'
UNKKEY = '~@@UNK@@~'


class Indexer():
    """ Indexer Class """

    def __init__(self, norm):
        self.key_to_idx = {}
        self.idx_to_key = {}
        self.norm = norm

    def add(self, key):
        """ Add """

        key = self._norm(key)

        if key not in self.key_to_idx:
            new_idx = len(self.key_to_idx)
            self.idx_to_key[new_idx] = key
            self.key_to_idx[key] = new_idx
            return new_idx
        return self.key_to_idx[key]

    def index_of(self, key):
        """ Get index of a key """

        key = self._norm(key)

        if key not in self.key_to_idx:
            return -1
        return self.key_to_idx[key]

    def key_of(self, index):
        """ Get the key of an index """

        if index not in self.idx_to_key:
            return -1
        return self.idx_to_key[index]

    def index(self, items, replace=UNKKEY):
        """ if norm is true, index will be by the lowercase string """

        if self.norm:
            items = [self._norm(item) for item in items]

        return [self.key_to_idx[item] if item in self.key_to_idx else
                self.key_to_idx[replace] for item in items]

    def index_with_replace(self, items, replace_list, rate=0.,
                           replace=UNKKEY):
        """ same as index but will replace any item in replace list with
        replace at given rate.
        The most likely use for this is randomly dropping singletons."""

        if self.norm:
            items = [self._norm(item) for item in items]
            replace_list = [self._norm(item) for item in replace_list]

        idxd = [self.key_to_idx[item] if item in self.key_to_idx else
                self.key_to_idx[replace] for item in items]

        if rate == 0.:
            return idxd

        for i, item in enumerate(items):
            if item in replace_list:
                if rate > np.random.random_sample():
                    idxd[i] = self.key_to_idx[replace]
        return idxd

    def rev_index(self, items):
        """ Assume only valid indexes """

        return [self.idx_to_key[item] for item in items]

    def _norm(self, key):
        if key in (UNKKEY, PADKEY):
            return key
        if not self.norm:
            return key
        return key.lower()

    def __len__(self):
        return len(self.key_to_idx)


def is_begin_tag(tag):
    """ Simple helper """
    return tag.startswith('B-')


# ------------
# CoNLL Utils
# ------------


def is_inside_tag(tag):
    """ Simple helper """
    return tag.startswith('I-')


def get_tag_type(tag):
    """ Simple helper """
    return tag.split('-')[-1]


def viterbi_decode(score, transition_params=None, transition_weights=None):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    Note that unlike beam search, the Viterbi algorithm makes a Markov
    assumption concerning state transitions. This allows us to
    consider all possible sequences efficiently using dynamic
    programming but is only appropriate for models like HMMs and CRFs.

    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      transition_weights: A [seq_len, num_tags, num_tags] tensor of state
        transition weights. The transition params are multiplied by these weights.
        To disable a transition from state i to state j, use negative infinity:
        `transition_weights[i][j] = -np.inf`.

    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
      viterbi_score: A float containing the score for the Viterbi sequence.

    """
    if transition_params is None:
        n_tags = score.shape[-1]
        transition_params = np.ones([n_tags, n_tags], dtype=np.float32)

    if transition_weights is None:
        n_tags = score.shape[-1]
        seq_len = score.shape[0]
        transition_weights = np.ones((seq_len, n_tags, n_tags), dtype=np.float)

    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params * transition_weights[t]
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score
