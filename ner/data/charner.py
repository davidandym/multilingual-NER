"""
Generator and Predictor classes for the byte-level CharNER model for NER.
"""


# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=invalid-name


import tensorflow as tf
import numpy as np

import ner.data.dataset as data
from ner.data.io import load_label_map
from ner.features import Features, Predictions
from ner.registry import Registries
from ner.utils import viterbi_decode


@Registries.train_generators.register("charner")
@Registries.test_generators.register("charner")
class CharNerModelGenerator(data.Generator):
    """ Generator for the CharNER model. The generator here is relatively
    simple. Each example is converted into a sequence of bytes and BIO-less
    tags for each byte. It is trained with an iid cross-entropy loss.
    For more details, see: https://www.aclweb.org/anthology/C16-1087/

    It would probably speed things up to split up sequences of bytes by a
    max-length parameters with some amount of overlap between splits, but I
    didn't do this and things ran in a decent amount of time.
    """
    def __init__(self, *, dataset, hp):
        data.Generator.__init__(self, dataset, hp)
        self.nobio_label_map = load_label_map(hp.nobio_label_map_path)

        # set the output dimensions based on the size of the label map.
        num_labels = len(self.nobio_label_map)
        hp.set_hparam('output_dim', num_labels)

    def iterate(self):
        for sentence in self.dataset.get_sentences():
            # tags do not have BIO info
            byte_list, tag_list = sentence.get_byte_and_tag_sequence()
            targets = [self.nobio_label_map[tag] for tag in tag_list]
            features = {
                Features.WINDOW_ID.value: [sentence.id],
                Features.INPUT_SEQUENCE_LENGTH.value: [len(byte_list)],
                Features.INPUT_SYMBOLS.value: byte_list
            }
            yield (features, targets)

    def estimator_params(self):
        return {}

    def datatypes(self):
        features = {
            Features.INPUT_SEQUENCE_LENGTH.value: tf.int64,
            Features.INPUT_SYMBOLS.value: tf.int64,
            Features.WINDOW_ID.value: tf.int64
        }

        return (features, tf.int64)

    def datashape(self):
        features = {
            Features.INPUT_SEQUENCE_LENGTH.value: [1],
            Features.INPUT_SYMBOLS.value: [None],
            Features.WINDOW_ID.value: 1
        }

        return (features, [None])



@Registries.predictors.register("charner")
class CharNerModelPredictor(data.Predictor):
    """ Gathers byte-level logit predictions, and converts them into word-level
    BIO tags.

    This is done using a viterbi decoder with transition matrices that
    constrain the output tags to be consistent at the word-level.
    For more details, see: https://www.aclweb.org/anthology/C16-1087/
    """
    def __init__(self, dataset, hp):
        data.Predictor.__init__(self, dataset, hp)
        self.nobio_label_map = load_label_map(hp.nobio_label_map_path)
        self.rev_label_map = {v: k for k, v in self.nobio_label_map.items()}

    def gather(self, predictions):
        sentences = self.dataset.get_sentences()

        for prediction in predictions:
            # sentence id's are just indexes.
            sentence_id = prediction[Predictions.WINDOW_ID.value][0]
            sentence = sentences[sentence_id]
            assert sentence_id == sentence.id

            seq_len = prediction[Predictions.LENGTH.value][0]
            logits = prediction[Predictions.RAW_TOKEN_SCORES.value][:seq_len]

            byte_seq, _ = sentence.get_byte_and_tag_sequence()
            transition_matrices = self.get_transition_matrices(byte_seq)

            decoded_tags, _ = viterbi_decode(
                score=logits,
                transition_weights=transition_matrices
            )

            pred_tags = self.conv_to_word_tags(decoded_tags, sentence.words)

            sentence_preds = []
            for i, word in enumerate(sentence.words):
                sentence_preds.append(
                    (word, sentence.tag_list[i], pred_tags[i])
                )

            self.sentence_predictions.append(sentence_preds)

    def get_transition_matrices(self, byte_seq):
        """ Get transition matrices used for word-level consistent decoding.
        For more information, see: https://www.aclweb.org/anthology/C16-1087/
        """
        nlabels = len(self.nobio_label_map)
        o_idx = self.nobio_label_map["O"]

        trans_matrices = []
        # the first matrix here get's thrown away
        trans_matr = np.zeros((nlabels, nlabels))
        trans_matrices += [trans_matr]
        for curb_idx in range(len(byte_seq) - 1):
            trans_matr = np.ones((nlabels, nlabels))
            trans_matr = trans_matr * -np.inf

            np.fill_diagonal(trans_matr, 0.)

            cur_byte = byte_seq[curb_idx]
            next_byte = byte_seq[curb_idx+1]

            if cur_byte == data.SPACE_BYTE:
                trans_matr[o_idx] = np.zeros(nlabels)
            elif next_byte == data.SPACE_BYTE:
                trans_matr[:, o_idx] = np.zeros(nlabels)

            trans_matrices += [trans_matr]
        return trans_matrices

    def get_tag(self, cur_tag, space_tag):
        """ Convert a nobio tag into a bio tag using previous space tag. """
        if cur_tag == self.nobio_label_map['O']:
            return 'O'
        if space_tag == self.nobio_label_map['O']:
            return f"B-{self.rev_label_map[cur_tag]}"
        return f"I-{self.rev_label_map[cur_tag]}"

    def conv_to_word_tags(self, tags, words):
        """ Convert byte-level tag sequences to word-level tags. """
        cur_byte = 0
        pred_tags = []

        for word in words:
            byte_tag = tags[cur_byte]
            prev_space_tag = tags[cur_byte-1] if cur_byte != 0 \
                                              else self.nobio_label_map['O']
            pred_tags.append(self.get_tag(byte_tag, prev_space_tag))
            cur_byte += len(bytes(word, encoding='utf-8')) + 1
        assert cur_byte == len(tags) + 1 # used all bytes. sanity
        return pred_tags
