"""Use Features to avoid hard-coding any feature names.

The Predictions `Enum` serves a similar purpose but for retrieving
predictions from the model.
"""


from enum import Enum


class Features(Enum):
    """All possible features that might be associated with an example"""

    # Sequence of FastText features.
    FASTTEXT_FEATURE_SEQUENCE = "fasttext_feat_seq"

    # A flat list of input symbols (e.g. subword indices)
    INPUT_SYMBOLS = "input_symbols"

    # The length of the input sequence. This is used for either a list
    # of input symbols (e.g. subword token indices) or a list of
    # feature vectors.
    INPUT_SEQUENCE_LENGTH = "input_seq_len"

    # A sequence of targets to predict. This doesn't necessarily have
    # the same length as the inputs.
    TARGET_SEQUENCE = "targets"

    # A sequence of targets offset to be used as teacher labels for teacher
    # forcing (only used for byte-to-span, really).
    TEACHER_TARGET_SEQUENCE = "teacher_targets"

    TARGET_SEQUENCE_LENGTH = "target_length"

    # A global identifier for a particular instance. For example, a
    # sentence ID.
    WINDOW_ID = "window_id"

    # The relative position of an example compared to other examples.
    REL_POS = "relative_position"

    # Indexed input words.
    INPUT_WORDS = "word_ids"

    # Indexed characters for each input word.
    INPUT_WORD_CHARS = "char_ids"

    # Lengths of each word, in characters
    INPUT_CHAR_LENGTHS = "char_lengths"

    # Bytes for each input word.
    INPUT_WORD_BYTES = "byte_ids"

    # Lengths of each word, in bytes
    INPUT_BYTE_LENGTHS = "byte_lengths"

    # Indexed subwords for each input word.
    INPUT_WORD_SUBWORDS = "subword_ids"

    # Lengths of each word, in subwords
    INPUT_SUBWORD_LENGTHS = "subword_lengths"


class Predictions(Enum):
    """ All possible predictions a model might produce """

    # Predicted named-entity labels
    TAGS = "y_hat"

    # Length of the prediction (to account for possible padding)
    LENGTH = "seq_length"

    RAW_TOKEN_SCORES = "token_logits"

    # A global identifier for a particular instance. For example, a
    # sentence ID.
    WINDOW_ID = "window_id"

    # The relative position of an example compared to other examples.
    REL_POS = "relative_position"
