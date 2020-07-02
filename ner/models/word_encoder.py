""" Embedding layers to encode tokens into vectors. """


# pylint: disable=too-few-public-methods,too-many-locals


import tensorflow as tf

from ner.models.model import Embedding
from ner.features import Features


class BasicWordEncoder(Embedding):
    """ Basic Word embedding methods.
    Allows embeddings of words using:
        - character features
        - byte features
        - subword features
        - word embeddings:
            - pretrained
            - randomly initialized
        - or any combination of the above, concatenated together.
    Currently, subtoken encodings are generated via a BiRNN
    Optionally if supplied -emb_enc_type cnn
    supports CNN subword encoder.
    """

    def __init__(self, hparams):
        Embedding.__init__(self, hparams)
        self.word_embedding_intializer = None
        self.word_embedding_shape = None
        self.scaffold = None

    def embed(self, *, inputs, params, is_training):
        """ build token embeddings, using some combination of subtoken and
        word-embeddings.
        """
        token_feats = []

        if 'bytes' in self.hp.features:
            keep_prob = self.hp.byte_enc_keep_prob if is_training else 1.
            if self.hp.emb_enc_type_cnn:
                byte_features = \
                    subtoken_cnn_encoder(
                        namespace='byte-cnn-encoder',
                        inputs=inputs[Features.INPUT_WORD_BYTES.value],
                        token_lengths=inputs[Features.INPUT_BYTE_LENGTHS.value],
                        padding_size=self.hp.byte_pad_length,
                        subtoken_vocab_size=256,
                        subtoken_embedding_size=self.hp.byte_emb_dim,
                        kernel_size=self.hp.emb_kernel_size,
                        num_filters=self.hp.emb_output_size)
            else:
                byte_features = \
                        subtoken_birnn_encoder(
                            namespace='byte-encoder',
                            inputs=inputs[Features.INPUT_WORD_BYTES.value],
                            token_lengths=inputs[Features.INPUT_BYTE_LENGTHS.value],
                            padding_size=self.hp.byte_pad_length,
                            subtoken_vocab_size=256,
                            num_layers=self.hp.byte_enc_layers,
                            hidden_size=self.hp.byte_enc_hdim,
                            keep_prob=keep_prob,
                            subtoken_embedding_size=self.hp.byte_emb_dim)
            token_feats += [byte_features]

        if 'chars' in self.hp.features:
            keep_prob = self.hp.byte_enc_keep_prob if is_training else 1.
            if self.hp.emb_enc_type_cnn:
                char_features = \
                    subtoken_cnn_encoder(
                        namespace='char-cnn-encoder',
                        inputs=inputs[Features.INPUT_WORD_CHARS.value],
                        token_lengths=inputs[Features.INPUT_CHAR_LENGTHS.value],
                        padding_size=self.hp.byte_pad_length,
                        subtoken_vocab_size=self.hp.char_voc_size,
                        subtoken_embedding_size=self.hp.byte_emb_dim,
                        kernel_size=self.hp.emb_kernel_size,
                        num_filters=self.hp.emb_output_size)
            else:
                char_features = \
                        subtoken_birnn_encoder(
                            namespace='char-encoder',
                            inputs=inputs[Features.INPUT_WORD_CHARS.value],
                            token_lengths=inputs[Features.INPUT_CHAR_LENGTHS.value],
                            padding_size=self.hp.byte_pad_length,
                            subtoken_vocab_size=self.hp.char_voc_size,
                            num_layers=self.hp.byte_enc_layers,
                            hidden_size=self.hp.byte_enc_hdim,
                            keep_prob=keep_prob,
                            subtoken_embedding_size=self.hp.byte_emb_dim)
            token_feats += [char_features]

        if 'subwords' in self.hp.features:
            keep_prob = self.hp.subword_enc_keep_prob if is_training else 1.
            if self.hp.emb_enc_type_cnn:
                subword_features = \
                    subtoken_cnn_encoder(
                        namespace='subword-cnn-encoder',
                        inputs=inputs[Features.INPUT_WORD_SUBWORDS.value],
                        padding_size=self.hp.subword_pad_length,
                        token_lengths=inputs[
                            Features.INPUT_SUBWORD_LENGTHS.value],
                        subtoken_vocab_size=self.hp.subword_voc_size,
                        subtoken_embedding_size=self.hp.byte_emb_dim,
                        kernel_size=self.hp.emb_kernel_size,
                        num_filters=self.hp.emb_output_size)
            else:
                subword_features = \
                        subtoken_birnn_encoder(
                            namespace='subword-encoder',
                            inputs=inputs[Features.INPUT_WORD_SUBWORDS.value],
                            token_lengths=inputs[
                                Features.INPUT_SUBWORD_LENGTHS.value],
                            padding_size=self.hp.subword_pad_length,
                            subtoken_vocab_size=self.hp.subword_voc_size,
                            num_layers=self.hp.subword_enc_layers,
                            hidden_size=self.hp.subword_enc_hdim,
                            keep_prob=keep_prob,
                            subtoken_embedding_size=self.hp.subword_emb_dim)
            token_feats += [subword_features]

        if 'words' in self.hp.features:
            # setup word-embedding matrix
            if self.hp.pretrained_embeddings:
                embs = params['word_embeddings']
                if embs is None:
                    raise Exception("No word embeddings passed in params!")
                self.word_embedding_intializer = tf.placeholder(
                    shape=embs.shape,
                    dtype=tf.float32
                )

                self.scaffold = tf.train.Scaffold(
                    init_feed_dict={self.word_embedding_intializer: embs}
                )
            else:
                self.word_embedding_shape = [self.hp.word_voc_size,
                                             self.hp.word_emb_dim]

            with tf.device('/cpu:0'):
                embedding = tf.get_variable(
                    "embed_mat",
                    shape=self.word_embedding_shape,
                    regularizer=None,
                    initializer=self.word_embedding_intializer
                )

                word_embeddings = tf.nn.embedding_lookup(
                    embedding,
                    inputs[Features.INPUT_WORDS.value]
                )

                token_feats += [word_embeddings]

        return tf.concat(token_feats, axis=-1) if len(token_feats) > 1 else token_feats[0]


def subtoken_birnn_encoder(*, namespace, inputs, token_lengths, padding_size,
                           subtoken_vocab_size, cell_type='LSTM', num_layers=1,
                           hidden_size=256, keep_prob=1.,
                           subtoken_embedding_size=256):
    """ Helper method for token encoding. Encodes tokens using a BiRNN over
    some subtoken pieces.
    Assumes random initialization for subtoken piece embeddings.
    """

    with tf.variable_scope(namespace):
        batch_size = tf.shape(inputs)[0]
        batch_len = tf.shape(token_lengths)[1]


        initializer = tf.initializers.truncated_normal(0.0, 0.001)
        embedding = tf.get_variable("embeddings",
                                    shape=[subtoken_vocab_size,
                                           subtoken_embedding_size],
                                    trainable=True,
                                    initializer=initializer)

        embedded = tf.nn.embedding_lookup(embedding, inputs)
        embedded = tf.reshape(embedded, [-1,
                                         padding_size,
                                         subtoken_embedding_size])

        lengths = tf.reshape(token_lengths, [batch_size * batch_len])

        def cell_fn():
            cell = getattr(tf.nn.rnn_cell, f'{cell_type}Cell')(hidden_size)
            return tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob)

        fw_cells = [cell_fn() for _ in range(num_layers)]
        bw_cells = [cell_fn() for _ in range(num_layers)]

        _, fwd, bwd = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw_cells,
            bw_cells,
            embedded,
            sequence_length=lengths,
            dtype=tf.float32)

        if isinstance(fwd, tuple):
            assert isinstance(bwd, tuple)
            states = []
            for state in fwd + bwd:
                if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
                    states.append(state.h)
                else:
                    states.append(state)

            net = tf.concat(states, axis=-1)
        else:
            raise ValueError(fwd)

    return tf.reshape(net, [batch_size, batch_len, hidden_size * 2 *
                            num_layers])


def subtoken_cnn_encoder(*, namespace, inputs, padding_size,
                         subtoken_vocab_size, token_lengths,
                         num_filters=30, kernel_size=3,
                         subtoken_embedding_size=256,
                         padding='same', activation='tanh', strides=1):
    """ Helper method for token encoding. Encodes tokens using a CNN over
    some subtoken pieces.
    Assumes random initialization for subtoken piece embeddings.
    """

    with tf.variable_scope(namespace):
        batch_size = tf.shape(inputs)[0]
        batch_len = tf.shape(token_lengths)[1]

        # Calculate embeddings
        initializer = tf.initializers.truncated_normal(0.0, 0.001)
        embedding = tf.get_variable("embeddings",
                                    shape=[subtoken_vocab_size,
                                           subtoken_embedding_size],
                                    trainable=True,
                                    initializer=initializer)

        embedded = tf.nn.embedding_lookup(embedding, inputs)

        embedded = tf.reshape(embedded, [-1,
                                         padding_size,
                                         subtoken_embedding_size])

        ## initialize CNN layers
        conv1d_layer = tf.keras.layers.Conv1D(filters=num_filters, \
                                        kernel_size=kernel_size, \
                                        padding=padding, activation=activation, strides=strides)

        ## Apply CNN layers
        conv1d_out = conv1d_layer.apply(embedded)
        max_out = tf.reduce_max(conv1d_out, axis=1)

        return tf.reshape(max_out, [batch_size, batch_len, num_filters])
