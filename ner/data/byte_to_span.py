"""
Generator and Predictor classes for the byte-level byte-to-span model for NER.
"""


# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=invalid-name


from math import floor

import tensorflow as tf
import numpy as np

import ner.data.dataset as data
from ner.features import Features, Predictions
from ner.registry import Registries
import ner.byte as byte


@Registries.train_generators.register("byte_to_span")
class Byte2SpanDSLevelTrainGenerator(data.Generator):
    """ Iterator for Byte2Span Model _training_ data.

    It iterates over the entire datasets worth of byte-sequences, creating
    independent segments at each point.
    """

    def __init__(self, *, dataset, hp):
        data.Generator.__init__(self, dataset, hp)

        self.window_size = hp.window_size
        self.stride = hp.stride
        self.spans = {}
        self.output_idxr = byte.seq2seq_indexer(self.window_size)

        self.duplicate_data = hp.duplicate_data

        self.byte_drop_rate = hp.byte_dropout_rate
        self.byte_drop_idx = 256

        complete_byte_seq, spans = self.dataset.get_dataset_as_byte_sequence_and_spans()
        self.complete_byte_seq = complete_byte_seq
        self.spans = spans

    def iterate(self):
        """ Iterate over consecutive windows of size self.window_size, with stride
        self.stride. Iterates over the entire dataset at once, so consecutive
        sentences share windows sometimes.
        """
        total_num_bytes = len(self.complete_byte_seq)
        num_steps = floor(total_num_bytes / self.stride)

        example_count = 0
        for i in range(num_steps):
            start = i * self.stride
            end = total_num_bytes \
                    if start + self.window_size > total_num_bytes \
                    else start + self.window_size

            byte_of_bytes = self.complete_byte_seq[start:end]
            spans = self.get_spans_within(segment_start=start, segment_end=end)

            segment_spans = []
            for span in spans:
                segment_spans.append(span.inside_segment_start(start))

            segment = byte.ByteSegment(_id=example_count,
                                       sent_id=0,
                                       wbytes=byte_of_bytes,
                                       spans=segment_spans,
                                       absolute_start=start)


            if self.byte_drop_rate >= 0.:
                seq = self.apply_drop_to_seq(segment.bytes)
                seq_no_drop = segment.bytes
            else:
                seq = segment.bytes
                seq_no_drop = None

            features = {
                Features.INPUT_SYMBOLS.value: seq,
                Features.INPUT_SEQUENCE_LENGTH.value: [len(seq)],
                Features.WINDOW_ID.value: [example_count],
                Features.REL_POS.value: [segment.absolute_start]
            }

            target, teacher = segment.build_gold_labels()

            target_idxd = [self.output_idxr[t] for t in target]
            teacher_idxd = [self.output_idxr[t] for t in teacher]

            targets = {
                Features.TARGET_SEQUENCE.value: target_idxd,
                Features.TEACHER_TARGET_SEQUENCE.value: teacher_idxd,
                Features.TARGET_SEQUENCE_LENGTH.value: [len(target_idxd)]
            }

            example_count += 1
            yield features, targets

            if self.duplicate_data:
                assert seq_no_drop is not None
                features = {
                    Features.INPUT_SYMBOLS.value: seq_no_drop,
                    Features.INPUT_SEQUENCE_LENGTH.value: [len(seq)],
                    Features.WINDOW_ID.value: [example_count],
                    Features.REL_POS.value: [segment.absolute_start]
                }

                yield features, targets

    def estimator_params(self):
        """ Return parameters for tf.Estimator class for training. """
        return {
            'output-voc-size': len(self.output_idxr),
            'go-idx': self.output_idxr['GO'],
            'stop-idx': self.output_idxr['STOP'],
        }

    def datashape(self):
        features = {
            Features.INPUT_SYMBOLS.value: [None],
            Features.INPUT_SEQUENCE_LENGTH.value: [1],
            Features.WINDOW_ID.value: 1,
            Features.REL_POS.value: 1,
        }
        targets = {
            Features.TARGET_SEQUENCE.value: [None],
            Features.TEACHER_TARGET_SEQUENCE.value: [None],
            Features.TARGET_SEQUENCE_LENGTH.value: [1]
        }
        return (features, targets)

    def datatypes(self):
        features = {
            Features.INPUT_SYMBOLS.value: tf.int64,
            Features.INPUT_SEQUENCE_LENGTH.value: tf.int64,
            Features.WINDOW_ID.value: tf.int64,
            Features.REL_POS.value: tf.int64,
        }
        targets = {
            Features.TARGET_SEQUENCE.value: tf.int64,
            Features.TEACHER_TARGET_SEQUENCE.value: tf.int64,
            Features.TARGET_SEQUENCE_LENGTH.value: tf.int64
        }
        return (features, targets)

    def get_spans_within(self, *, segment_start, segment_end):
        """ Get all spans inside a certain window """
        spans = []

        for i in range(segment_start, segment_end):
            if i in self.spans:
                span = self.spans[i]
                if span.contained_in(seg_start=segment_start, seg_end=segment_end):
                    spans.append(span)

        return spans

    def apply_drop_to_seq(self, seq):
        """ Byte-dropout. Drop a certain number of bytes in the sequence.
        From https://arxiv.org/abs/1512.00103
        """
        out_seq = []
        for byt in seq:
            # just some sanity checking
            assert byt != self.byte_drop_idx
            r = np.random.rand()
            if r <= self.byte_drop_rate:
                out_seq.append(self.byte_drop_idx)
            else:
                out_seq.append(byt)
        assert len(out_seq) == len(seq)
        return out_seq


@Registries.test_generators.register("byte_to_span")
class Byte2SpanDSLevelTestGenerator(data.Generator):
    """ Iterator for Byte2Span Model _test_ data.

    It iterates over the entire datasets worth of byte-sequences, creating
    independent segments with overlaps of window_size, to be stitched together
    for prediction.
    """
    def __init__(self, *, dataset, hp):
        data.Generator.__init__(self, dataset, hp)
        self.window_size = hp.window_size
        self.overlap = hp.test_overlap
        self.stride = self.window_size - self.overlap
        self.output_idxr = byte.seq2seq_indexer(self.window_size)

        complete_byte_seq, spans = self.dataset.get_dataset_as_byte_sequence_and_spans()
        self.complete_byte_seq = complete_byte_seq
        self.spans = spans
        self.segment_map = {}

    def iterate(self):
        num_steps = self.num_steps()
        total_num_bytes = len(self.complete_byte_seq)

        tf.logging.info("{} Test Segments To Be Generated!".format(num_steps))

        for i in range(num_steps):
            start = i * self.stride
            end = total_num_bytes \
                    if start + self.window_size > total_num_bytes \
                    else start + self.window_size

            byte_of_bytes = self.complete_byte_seq[start:end]
            spans = self.get_spans_within(segment_start=start, segment_end=end)

            segment_spans = []
            for span in spans:
                segment_spans.append(span.inside_segment_start(start))

            segment = byte.ByteSegment(_id=i,
                                       sent_id=0,
                                       wbytes=byte_of_bytes,
                                       spans=segment_spans,
                                       absolute_start=start)

            self.segment_map[i] = segment

            features = {
                Features.INPUT_SYMBOLS.value: segment.bytes,
                Features.INPUT_SEQUENCE_LENGTH.value: [len(segment.bytes)],
                Features.WINDOW_ID.value: [i],
                Features.REL_POS.value: [segment.absolute_start]
            }

            target, teacher = segment.build_gold_labels()

            target_idxd = [self.output_idxr[t] for t in target]
            teacher_idxd = [self.output_idxr[t] for t in teacher]

            targets = {
                Features.TARGET_SEQUENCE.value: target_idxd,
                Features.TEACHER_TARGET_SEQUENCE.value: teacher_idxd,
                Features.TARGET_SEQUENCE_LENGTH.value: [len(target_idxd)]
            }

            yield features, targets

    def estimator_params(self):
        """ Return parameters for tf.Estimator class for prediction. """
        return {
            'output-voc-size': len(self.output_idxr),
            'go-idx': self.output_idxr['GO'],
            'stop-idx': self.output_idxr['STOP'],
        }

    def datashape(self):
        features = {
            Features.INPUT_SYMBOLS.value: [None],
            Features.INPUT_SEQUENCE_LENGTH.value: [1],
            Features.WINDOW_ID.value: 1,
            Features.REL_POS.value: 1,
        }
        targets = {
            Features.TARGET_SEQUENCE.value: [None],
            Features.TEACHER_TARGET_SEQUENCE.value: [None],
            Features.TARGET_SEQUENCE_LENGTH.value: [1]
        }
        return (features, targets)

    def datatypes(self):
        features = {
            Features.INPUT_SYMBOLS.value: tf.int64,
            Features.INPUT_SEQUENCE_LENGTH.value: tf.int64,
            Features.WINDOW_ID.value: tf.int64,
            Features.REL_POS.value: tf.int64,
        }
        targets = {
            Features.TARGET_SEQUENCE.value: tf.int64,
            Features.TEACHER_TARGET_SEQUENCE.value: tf.int64,
            Features.TARGET_SEQUENCE_LENGTH.value: tf.int64
        }
        return (features, targets)

    def get_spans_within(self, *, segment_start, segment_end):
        """ Get all spans inside a certain window """
        spans = []

        for i in range(segment_start, segment_end):
            if i in self.spans:
                span = self.spans[i]
                if span.contained_in(seg_start=segment_start, seg_end=segment_end):
                    spans.append(span)

        return spans

    def num_steps(self):
        """ Number of steps it takes to iterate over all segments of the
        dataset.
        """
        seq_len = len(self.complete_byte_seq)
        max_start = max(seq_len - self.window_size, 0)

        if max_start % self.stride == 0:
            num_steps = (max_start / self.stride) + 1
        else:
            num_steps = floor(max_start/self.stride) + 2

        return int(num_steps)

    def get_segment_map(self):
        """ Helper. """
        return self.segment_map


@Registries.predictors.register("byte_to_span")
class Byte2SpanPredictor(data.Predictor):
    """ Inference for a Byte2Span model.

        We're assuming that inference was done at a "dataset-level". I.e. we're
        recieiving segments whose absolute position feature indicates that
        segments absolute position within the byte sequence of the _entire_
        dataset.

        So, in order to do inference, this class decodes the spans from each
        segment's predicted sequence, and then resolves conflicts etc. and then
        decodes each word label by deciding whether or not that word falls
        within a span.

        This is a rather messy and complex little class. I'm sorry.
    """

    def __init__(self, dataset, hp):
        data.Predictor.__init__(self, dataset, hp)

        self.rev_idxr = byte.seq2seq_rev_indexer(
            byte.seq2seq_indexer(hp.window_size)
        )
        self.test_overlap = hp.test_overlap

        self.all_predicted_spans = []
        self.filtered_predicted_spans = {}

    def gather(self, predictions):
        """ infer and store predictions.

            Predictions, in this case, should be the predictions of a byte2span
            model (sequences of span predictions, over an entire dataset)
        """

        self.segment_prediction_map = {}

        for prediction in predictions:
            segment_id = prediction[Predictions.WINDOW_ID.value][0]
            self.segment_prediction_map[segment_id] = prediction

            p_seq_len = prediction[Predictions.LENGTH.value]
            p_seq = prediction[Predictions.TAGS.value][:p_seq_len]
            p_spans = self.get_spans_from_seq(p_seq)

            for p_span in p_spans:
                self.all_predicted_spans += [
                    p_span.outside_segment_start(prediction[Predictions.REL_POS.value][0])
                ]

        tf.logging.info("{} Total spans predicted!".format(len(self.all_predicted_spans)))
        self.filter_predicted_spans()
        tf.logging.info("{} Filtered spans predicted!".format(len(self.filtered_predicted_spans)))

        cur_abs_idx = 0
        prev_tag = 'O'

        for sentence in self.dataset.get_sentences():

            sentence_predicted_tags = []

            for i, word in enumerate(sentence.word_list):

                word_size = len(bytes(word, encoding='utf-8'))
                word_start = cur_abs_idx
                word_end = cur_abs_idx + word_size

                word_span = self.find_span_for_word(word_start, word_end)

                if word_span is None:
                    sentence_predicted_tags.append((word,
                                                    sentence.tag_list[i],
                                                    'O'))
                    prev_tag = 'O'
                else:
                    word_type = word_span.tag
                    if word_span.start >= word_start - 1:
                        p_tag = 'B-{}'.format(word_type)
                    elif word_type == prev_tag:
                        p_tag = 'I-{}'.format(word_type)
                    else:
                        p_tag = 'B-{}'.format(word_type)

                    prev_tag = word_type
                    sentence_predicted_tags.append((word,
                                                    sentence.tag_list[i],
                                                    p_tag))


                cur_abs_idx += word_size + 1

            assert len(sentence_predicted_tags) == len(sentence.word_list)
            self.sentence_predictions.append(sentence_predicted_tags)

            prev_tag = 'O'

        assert len(self.sentence_predictions) == len(self.dataset.get_sentences())

    def find_span_for_word(self, word_abs_start, word_abs_end):
        """ Find a span that a word falls in. If no span exists, return none.
        To avoid searching through every span, we iterate backwards from the
        word start position to find a span that starts at that position. Once
        we find one (we only need to evaluate on the first span found), we
        check to see if the word falls within that span."""
        potential_span = None

        for i in range(word_abs_start, -1, -1):
            if i in self.filtered_predicted_spans:
                potential_span = self.filtered_predicted_spans[i]
                break

        if potential_span is not None:
            if word_abs_end <= potential_span.start + potential_span.length:
                return potential_span

        return None

    def get_spans_from_seq(self, seq):
        """ Recover predicted spans from a predicted sequence. """
        spans = []

        cur_idx = 0
        cur_tag = self.rev_idxr[seq[cur_idx]]

        while cur_tag != "STOP":
            if cur_idx + 3 >= len(seq):
                break

            len_tag = self.rev_idxr[seq[cur_idx+1]]
            typ_tag = self.rev_idxr[seq[cur_idx+2]]

            if self.is_start(cur_tag) and self.is_length(len_tag) and self.is_type(typ_tag):
                spans.append(byte.ByteSpan(start=self.get_num(cur_tag),
                                           length=self.get_num(len_tag),
                                           tag=typ_tag))


                cur_idx += 3
                cur_tag = self.rev_idxr[seq[cur_idx]]
            else:
                cur_idx += 1
                cur_tag = self.rev_idxr[seq[cur_idx]]

        return spans

    def filter_predicted_spans(self):
        """ Filter predicted spans to non-conflicting (overlapping) spans. """
        pred_spans = sorted(self.all_predicted_spans, key=lambda x: x.start)

        cur_idx = 0
        while cur_idx < len(pred_spans):
            cur_span = pred_spans[cur_idx]

            # iterate over potential conflicts, and resolve
            for i in range(cur_idx+1, len(pred_spans)):
                compare_span = pred_spans[i]
                # if the next span starts before current span ends, conflict
                if compare_span.start < cur_span.start + cur_span.length:
                    # In the case of conflict, select the span which started
                    # earliest in it's respective segment. This heuristic comes
                    # from correspondence with the authors of byte-to-span.
                    if compare_span.seg_rel_start < cur_span.seg_rel_start:
                        cur_span = compare_span
                    cur_idx += 1
                # if no conflict, break out and move onto next span
                else:
                    break
            # presumably, there is no conflict anymore with the current span
            self.filtered_predicted_spans[cur_span.start] = cur_span
            cur_idx += 1

    @classmethod
    def is_length(cls, label):
        """ Helper. """
        return label.startswith("L:")

    @classmethod
    def is_start(cls, label):
        """ Helper. """
        return label.startswith("S:")

    @classmethod
    def get_num(cls, label):
        """ Helper. """
        return int(label[2:])

    @classmethod
    def is_type(cls, label):
        """ Helper. """
        return not label.startswith("L:") and \
               not label.startswith("S:") and \
               not label == "STOP" and \
               not label == "GO"
