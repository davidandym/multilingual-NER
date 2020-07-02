""" Implementation of Byte-to-span model described in
        https://www.aclweb.org/anthology/N16-1155/
"""


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=invalid-name


import tensorflow as tf

from ner.models.model import Model
from ner.features import Features, Predictions
from ner.registry import Registries
from ner.hparams import HParams
from ner.optim import get_training_op


# ------------------------
# Hyperparameter Defaults
# ------------------------


# Byte-to-Span default with momentum
@Registries.hparams.register
def byte_to_span_momentum():
    return HParams(
        batch_size=128,
        shuffle_buffer_size=40000,
        window_size=60,
        test_overlap=30,
        stride=1,
        duplicate_data=False,
        byte_dropout_rate=0.3,
        byte_emb_size=256,
        cell_type='LSTM',
        rnn_num_layers=4,
        rnn_hidden_size=320,
        rnn_keep_prob=0.8,
        output_embedding_size=320,
        optimizer='momentum',
        momentum=0.9,
        gradient_clip_norm=5,
        learning_rate=0.001,
        learning_rate_schedule='cosine_decay',
        first_decay_steps=1000
    )


# ------------
# Model Class
# ------------


@Registries.models.register
class ByteToSpan(Model):
    """ Byte-to-span model.

    Runs an RNN over input bytes, and then uses an RNN decoder to predict
    labeled spans over the input.
    """

    def __init__(self, hparams):
        Model.__init__(self, hparams)

    def body(self, *, inputs, params, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        x = inputs[Features.INPUT_SYMBOLS.value]
        lengths = tf.reshape(
            inputs[Features.INPUT_SEQUENCE_LENGTH.value],
            [-1]
        )

        embeddings = tf.get_variable("byte_emb",
                                     shape=[257, self.hp.byte_emb_size],
                                     trainable=True)

        x = tf.nn.embedding_lookup(embeddings, x)
        encodings = rnn(x, is_training, lengths=lengths, hparams=self.hp)
        return encodings


    def loss(self, *, predictions, features, targets, is_training):
        # pulling out values
        batch_size = tf.cast(
            tf.shape(features[Features.INPUT_SYMBOLS.value])[0],
            tf.float32
        )
        target_labels = targets[Features.TARGET_SEQUENCE.value]
        target_lengths = tf.reshape(
            targets[Features.TARGET_SEQUENCE_LENGTH.value],
            [-1]
        )
        # compute loss
        target_mask = tf.sequence_mask(target_lengths, dtype=tf.float32)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels,
            logits=predictions
        )
        train_loss = (tf.reduce_sum(crossent*target_mask) / batch_size)
        return train_loss


    def get_model_fn(self, model_dir=None):

        def fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            batch_size = tf.cast(
                tf.shape(features[Features.INPUT_SYMBOLS.value])[0],
                tf.float32
            )

            encodings = self.body(inputs=features, params=params, mode=mode)
            if mode == tf.estimator.ModeKeys.PREDICT:
                target_lengths = 30
                target_inputs = None
            else:
                target_lengths = tf.reshape(
                    labels[Features.TARGET_SEQUENCE_LENGTH.value],
                    [-1]
                )
                target_inputs = labels[Features.TEACHER_TARGET_SEQUENCE.value]


            logits, out_lengths = rnn_decoder(encodings,
                                              is_training,
                                              batch_size,
                                              params['output-voc-size'],
                                              params['go-idx'],
                                              params['stop-idx'],
                                              self.hp,
                                              target_inputs=target_inputs,
                                              target_lens=target_lengths)


            if mode == tf.estimator.ModeKeys.TRAIN:

                loss = self.loss(predictions=logits,
                                 features=features,
                                 targets=labels,
                                 is_training=is_training)

                train_op = get_training_op(loss, self.hp)

                return tf.estimator.EstimatorSpec(mode,
                                                  loss=loss,
                                                  train_op=train_op)

            if mode == tf.estimator.ModeKeys.PREDICT:

                predictions = tf.argmax(logits, axis=-1)
                out = {
                    Predictions.TAGS.value      : predictions,
                    Predictions.LENGTH.value    : out_lengths,
                    Predictions.WINDOW_ID.value : features[Features.WINDOW_ID.value],
                    Predictions.REL_POS.value   : features[Features.REL_POS.value]
                }

                return tf.estimator.EstimatorSpec(mode, predictions=out)

            if mode == tf.estimator.ModeKeys.EVAL:

                loss = self.loss(predictions=logits,
                                 features=features,
                                 targets=labels,
                                 is_training=is_training)

                predictions = tf.argmax(logits, axis=-1)
                target_mask = tf.sequence_mask(target_lengths,
                                               dtype=tf.float32)
                eval_metric_ops = {
                    'accuracy': tf.metrics.accuracy(
                        labels[Features.TARGET_SEQUENCE.value],
                        predictions,
                        weights=target_mask
                    )
                }

                return tf.estimator.EstimatorSpec(mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metric_ops)

        return fn


# -----------------------
# Model Function Helpers
# -----------------------


def rnn_decoder(input_encoding, is_training, batch_size,
                output_vocab_size, start_idx, stop_idx,
                hparams, target_inputs=None, target_lens=None,
                max_decode_iters=None):
    """ sequence decoder """
    def cell_fn():
        if is_training:
            rnn_keep_prob = hparams.rnn_keep_prob
        else:
            rnn_keep_prob = 1.

        cell = getattr(tf.nn.rnn_cell, hparams.cell_type + 'Cell')(
            hparams.rnn_hidden_size,
            initializer=tf.initializers.random_uniform(minval=-0.08, maxval=0.08)
        )
        return tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=rnn_keep_prob,
                                             output_keep_prob=rnn_keep_prob,
                                             state_keep_prob=1.0)

    cells = [cell_fn() for _ in range(hparams.rnn_num_layers)]

    decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

    output_embeddings = tf.get_variable("output_embedding",
                                        shape=[output_vocab_size,
                                               hparams.output_embedding_size],
                                        trainable=True)
    projection_layer = tf.layers.Dense(output_vocab_size, use_bias=False)

    if is_training:
        assert target_lens is not None
        assert target_inputs is not None
        tlens = tf.cast(target_lens, tf.int32)
        target_output_embedded = tf.nn.embedding_lookup(output_embeddings,
                                                        target_inputs)
        helper = tf.contrib.seq2seq.TrainingHelper(target_output_embedded,
                                                   tlens)
    else:
        starts = tf.fill([batch_size], start_idx)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(output_embeddings,
                                                          starts,
                                                          stop_idx)

    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                              input_encoding,
                                              output_layer=projection_layer)

    outputs, _, lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_decode_iters
    )

    logits = outputs.rnn_output
    return logits, lengths


def rnn(inputs, is_training, lengths=None, hparams=None):
    """ rnn encoder. Runs a single direction multi-layer RNN over input, and
    returns the single last hidden state.
    """

    def cell_fn():
        if is_training:
            rnn_keep_prob = hparams.rnn_keep_prob
        else:
            rnn_keep_prob = 1.

        cell = getattr(tf.nn.rnn_cell, hparams.cell_type + 'Cell')(
            hparams.rnn_hidden_size,
            initializer=tf.initializers.random_uniform(minval=-0.08, maxval=0.08)
        )
        return tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=rnn_keep_prob,
                                             output_keep_prob=rnn_keep_prob,
                                             state_keep_prob=1.0)

    cells = [cell_fn() for _ in range(hparams.rnn_num_layers)]

    encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

    _, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                         inputs,
                                         sequence_length=lengths,
                                         dtype=tf.float32)
    return encoder_state
