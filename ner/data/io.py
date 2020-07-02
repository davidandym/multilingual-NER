"""
    Basic NER Data processing
      - Word and character / byte level pre-processing and data generators
      - Dataset loader
"""


# pylint: disable=invalid-name


import pickle

import tensorflow as tf
import sentencepiece as spm


def load_conll_data(conll_fname, max_sentence_len=0):
    """ Opens a conll file and splits it up into a list of sentences.
    Each sentence is a list of fields.
    Each list of fields represents one line of ConLL data,
    split up into the different items in each line
    [Token, POS, Coarse POS, ??, Tag]
    """

    word_count = 0
    doc_count = 0
    sentences = []
    curr_sentence = []
    with open(conll_fname, 'r', encoding='utf-8') as infile:
        for line in infile:
            stripped = line.strip()
            if stripped.startswith('-DOCSTART-'):
                doc_count += 1
                continue
            if stripped == "":
                if curr_sentence and \
                    (len(curr_sentence) < max_sentence_len or max_sentence_len <= 0):
                    sentences.append(curr_sentence)
                curr_sentence = []
            elif stripped != "":
                fields = stripped.split()
                curr_sentence.append(fields)
                word_count += 1

    if curr_sentence and \
        (len(curr_sentence) < max_sentence_len or max_sentence_len <= 0):
        sentences.append(curr_sentence)

    return sentences


def load_label_map(fname):
    """ Simple pickle load of a file. """
    m = pickle.load(open(fname, 'rb'))
    return m


def get_vocabs(vocab_file, subword_file=None):
    """ Load vocabs and embeddings, including sentencepiece model. """
    vocabs = pickle.load(open(vocab_file, 'rb'))
    subword_model = load_subword_model(subword_file)
    vocabs['subword-sp-model'] = subword_model
    return vocabs


def load_subword_model(subword_file):
    if subword_file != "" and subword_file is not None:
        sp = spm.SentencePieceProcessor()
        sp.Load(subword_file)
        tf.logging.info(('Loading Subword Sentence Piece Model!\n'
                         f'Subword Vocabulary Size: {len(sp)}'))
        return sp
    return None
