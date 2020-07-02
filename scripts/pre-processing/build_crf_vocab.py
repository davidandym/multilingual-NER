#! /usr/bin/env python
""" Builds a vocab object required for a word-based or character-based CRF. """


# pylint: disable=invalid-name
# pylint: disable=too-many-locals


import pickle
import argparse as ap
from io import open
from collections import defaultdict

import numpy as np

from ner.utils import PADKEY, UNKKEY, Indexer
from ner.data.io import load_conll_data


def get_args():
    """ Vocab args """
    p = ap.ArgumentParser()

    p.add_argument('--train-file', default=None, type=str,
                   help=('build vocab off of this'))
    p.add_argument('--test-files', default=None, type=str, nargs='+',
                   help=('files to keep pretrained embeddings of'))
    p.add_argument('--embedding-file', default=None, type=str,
                   help=('pretrained embeddings'))

    p.add_argument('--word-index', default=0, type=int)
    p.add_argument('--label-index', default=1, type=int)

    p.add_argument('--vocab-file', default='vocab.pickle', type=str)

    return p.parse_args()

def get_word_counts(args):
    """ Get word, label, and character counts from datasets.
    Collects word counts for words in train and test sets.
    """
    train_word_counts = defaultdict(int)
    test_word_counts = defaultdict(int)
    train_labels = defaultdict(int)
    char_counts = defaultdict(int)

    n_sent = 0
    for sentence in load_conll_data(args.train_file):
        n_sent += 1
        for fields in sentence:
            word = fields[args.word_index]
            for char in word:
                char_counts[char] += 1
            word = word.lower()
            train_word_counts[word] += 1

            label = fields[args.label_index]
            train_labels[label] += 1

    print("Read {} sentences".format(n_sent))
    print("{} words".format(len(train_word_counts)))
    label_list = list(train_labels.keys())
    print("{} labels found: {}".format(len(label_list), label_list))

    if args.test_files is not None:
        n_sent = 0
        for test_file in args.test_files:
            for sentence in load_conll_data(test_file):
                n_sent += 1
                for fields in sentence:
                    word = fields[args.word_index].lower()
                    if word not in train_word_counts:
                        test_word_counts[word] += 1

        print(f'Read {n_sent} test sentences')
        print(f'{len(test_word_counts)} Test words not appearing in Train data')

    return train_word_counts, test_word_counts, train_labels, char_counts

def read_embeddings(args, train_wc, test_wc, encoding='utf-8'):
    """ Read in a set of pretrained embeddings from an embedding file. """
    def random_vec(dim):
        return np.random.normal(size=dim)

    emb_f = open(args.embedding_file, 'r', encoding=encoding)

    indexer = Indexer(True)
    vectors = []
    first = True

    pretrained_train = 0
    pretrained_test = 0

    for line in emb_f:
        if first:
            first = False
            # if the first line looks like metadata, skip it. 5 is arbitrary.
            if len(line.strip().split()) < 5:
                continue

        if line.strip() != "":
            space_idx = line.find(' ')
            key = line[:space_idx]
            if key in train_wc:
                pretrained_train += 1
                numbers = line[space_idx+1:]
                indexer.add(key)
                float_numbers = [float(number_str) \
                                 for number_str in numbers.split()]
                vector = np.array(float_numbers)
                vectors.append(vector)
            elif key in test_wc:
                pretrained_test += 1
                numbers = line[space_idx+1:]
                indexer.add(key)
                float_numbers = [float(number_str) \
                                 for number_str in numbers.split()]
                vector = np.array(float_numbers)
                vectors.append(vector)
    dim = vectors[0].shape[0]
    emb_f.close()

    print((f'Read in {len(indexer)} pretrained vectors '
           f'of dim {vectors[0].shape[0]}'))
    print(f'Read in {pretrained_train} pretrained vectors from train')
    print(f'Read in {pretrained_test} pretrained vectors from test')

    ext_count = 0
    for word in train_wc:
        if indexer.index_of(word) == -1:
            ext_count += 1
            indexer.add(word)
            vectors.append(random_vec(dim))

    print(f'Added {ext_count} non-pretrained vocab words')

    indexer.add(UNKKEY)
    vectors.append(random_vec(dim))
    indexer.add(PADKEY)
    vectors.append(random_vec(dim))

    print("Final embedding init dim 0: {}".format(len(vectors)))

    assert len(indexer) == len(vectors)
    return indexer, np.array(vectors, ndmin=2)

def build_indexer(counts, norm=True, add_special=True):
    """ Builds a default Indexer from token counts.
    norm => all items will be lowercased
    add_special => add PAD and UNK tokens
    """
    indexer = Indexer(norm)

    for item in counts:
        indexer.add(item)

    if add_special:
        indexer.add(PADKEY)
        indexer.add(UNKKEY)

    return indexer

def get_singletons(counts):
    """ Return a dict of items which only appear once in counts.
    You might want to handle these specially, i.e. dropping 50% of singletones
    during training.
    """
    singletons = {}
    for item, count in counts.items():
        if count == 1:
            singletons[item] = True
    print(f'counted {len(singletons)} singletons')
    return singletons

def main(args):
    """ Build vocabs, save them in output pickle file """
    vocab = {}

    train_wc, test_wc, train_lbl, train_char_counts = get_word_counts(args)

    if args.embedding_file:
        word_indexer, vectors = read_embeddings(args, train_wc, test_wc)
        vocab['word-embeddings'] = vectors
        vocab['word'] = word_indexer
    else:
        vocab['word-embeddings'] = None
        word_indexer = build_indexer(train_wc, norm=True, add_special=True)
        vocab['word'] = word_indexer

    vocab['chars'] = build_indexer(train_char_counts,
                                   norm=False,
                                   add_special=True)
    vocab['label'] = build_indexer(train_lbl,
                                   norm=False,
                                   add_special=False)
    vocab['singletons'] = get_singletons(train_wc)

    pickle.dump(vocab, open(args.vocab_file, 'wb'))

if __name__ == '__main__':
    ARGS = get_args()
    main(ARGS)
