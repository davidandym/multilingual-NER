""" Some NER specific byte-processing helpers and classes.

    Includes:
        - byte2span output vocab maps
        - sentence to byte-sequence method
        - span building helper
        - ByteSpan class
        - ByteSegment class - more or less specific to the B2S model
"""


# pylint: disable=unsupported-assignment-operation,invalid-name


from ner.utils import is_inside_tag
from ner.utils import get_tag_type


# ---------------
# Generic Helper
# ---------------


def convert_sentence_into_byte_sequence(words, tags, space_idx=32, other='O'):
    """ Convert a list of words and their tags into a sequence of bytes, and
    the corresponding tag of each byte.
    """
    byte_list = []
    tag_list = []

    for word_index, (word, tag) in enumerate(zip(words, tags)):
        tag_type = get_tag_type(tag)

        if is_inside_tag(tag) and word_index > 0:
            byte_list += [space_idx]
            tag_list += [tag_type]
        elif word_index > 0:
            byte_list += [space_idx]
            tag_list += [other]

        b_seq = bytes(word, encoding='utf-8')

        nbytes = len(b_seq)
        byte_list += b_seq
        tag_list += [tag_type] * nbytes

    assert len(byte_list) == len(tag_list)
    return byte_list, tag_list


# ---------------------
# Byte-to-Span Indexer
# ---------------------


def seq2seq_indexer(max_segment_length):
    """ hard-coded b2s model output vocabulary """
    outputs = {
        'O': 0,
        'PER': 1,
        'ORG': 2,
        'GPE': 3,
        'LOC': 4,
        'MISC': 5,
        'FAC': 6,
        'TTL': 7,
        'MISSING': 8
    }

    outputs['STOP'] = len(outputs)
    outputs['GO'] = len(outputs)

    for i in range(max_segment_length):
        outputs["L:{}".format(i+1)] = len(outputs)
        outputs["S:{}".format(i)] = len(outputs)

    return outputs


def seq2seq_rev_indexer(output_idxr):
    """ b2s model inverse output vocabulary """
    rev_map = {}

    for key, value in output_idxr.items():
        rev_map[value] = key
    return rev_map


# ----------------------------------
# Byte-to-Span builders and classes
# ----------------------------------


def build_spans_from_tags(tags, other='O'):
    """ Take in a sequence of byte-tags, and returns a list of spans contained
    within the sequence.
    """
    spans = []
    cur_span = None
    cur_tag = other

    for i, tag in enumerate(tags):
        if tag == other and cur_span is not None:
            spans += [ByteSpan(start=cur_span['start'],
                               length=cur_span['len'],
                               tag=cur_span['type'])]
            cur_span = None
            cur_tag = other
        elif tag != cur_tag and cur_span is not None:
            spans += [cur_span]
            cur_tag = tag
            cur_span = {'type': tag, 'start': i, 'len': 1}
        elif tag != other and cur_span is None:
            cur_span = {'type': tag, 'start': i, 'len': 1}
            cur_tag = tag
        elif tag == cur_tag and cur_span is not None:
            cur_span['len'] += 1

    if cur_span is not None:
        spans += [ByteSpan(start=cur_span['start'],
                           length=cur_span['len'],
                           tag=cur_span['type'])]

    return spans


class ByteSpan():
    """ Span, representing some set of consecutive bytes that have some label
    over them. All a span is made up of is a start pos, length, and a
    label-type.
    """

    def __init__(self, *, start, length, tag, seg_rel_start=None):
        self.start = start
        self.length = length
        self.tag = tag
        self.seg_rel_start = seg_rel_start

    def contained_in(self, *, seg_start, seg_end):
        """ quickly determine whether or not a span is contained within a given
        segment.
        """
        return self.start >= seg_start and self.start + self.length <= seg_end

    def inside_segment_start(self, segment_start):
        """ Create a new span for a specific segment.
        This pass in the absolute segment start and the new span will
        contain it's relative start position within that segment.
        """
        new_start = self.start - segment_start
        return ByteSpan(start=new_start,
                        length=self.length,
                        tag=self.tag)

    def outside_segment_start(self, segment_start):
        """ Create a new span from a specific segment.
        Takes in the absolute start position of the segment this span was
        contained in, and returns a new span with the start updated.
        """
        new_start = self.start + segment_start
        return ByteSpan(start=new_start,
                        length=self.length,
                        tag=self.tag,
                        seg_rel_start=self.start)

    def __repr__(self):
        return "ByteSpan(S: {}, L: {}, T: {})".format(self.start,
                                                      self.length,
                                                      self.tag)

    def __eq__(self, other):
        if isinstance(other, ByteSpan):
            return self.start == other.start and \
                   self.length == other.length and \
                   self.tag == other.tag
        return False


class ByteSegment():
    """ A segment of bytes. This is the object that the byte2span model uses.
    It's composed of:
        - sentence_id and absolute_start for peicing segments back together,
          post inference
        - bytes, the input to the model
        - spans, what generally need to be predicted
    """

    def __init__(self, *, _id, sent_id, wbytes, spans, absolute_start):
        self.id = _id
        self.sent_id = sent_id
        self.bytes = wbytes
        self.spans = spans
        self.absolute_start = absolute_start

    def build_gold_labels(self, reverse=True):
        """ Build the teacher and target output sequences of this segment. """
        target_seq = []
        teacher_seq = ['GO']

        gold_spans = sorted(self.spans,
                            key=lambda x: x.start,
                            reverse=reverse)

        r = range(len(gold_spans))

        for span_idx in r:
            span = gold_spans[span_idx]
            start = "S:{}".format(span.start)
            length = "L:{}".format(span.length)
            tag_type = span.tag

            target_seq += [start, length, tag_type]
            teacher_seq += [start, length, tag_type]

        target_seq += ['STOP']

        assert len(target_seq) == len(teacher_seq)
        assert teacher_seq[0] == 'GO'
        assert teacher_seq[0] != 'STOP'
        assert target_seq[-1] == 'STOP'
        assert target_seq[-1] != 'GO'

        return target_seq, teacher_seq

    def __repr__(self):
        target_seq, _ = self.build_gold_labels()
        return ("Segment(Bytes: {}\n"
                "Spans: {}\n"
                "Sequence: {}\n").format(self.bytes, self.spans, target_seq)
