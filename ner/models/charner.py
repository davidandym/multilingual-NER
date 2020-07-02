""" Implementation of CharNER model described in
        https://www.aclweb.org/anthology/C16-1087
"""


# pylint: disable=invalid-name


import tensorflow as tf

from ner.hparams import HParams
from ner.models.model import Model
from ner.features import Features, Predictions
from ner.registry import Registries
from ner.optim import get_training_op


@Registries.hparams.register
def charner_default():
    return HParams(
        shuffle_buffer_size=40000,
        batch_size=32,
        output_dim=0,
        birnn_layers=5,
        birnn_dim=128,
        dropout_keep_prob=[0.5, 0.5, 0.5, 0.5, 0.2],
        optimizer='adam',
        beta1=0.9,
        beta2=0.999,
        use_ema=False,
        learning_rate=0.001,
        emb_dim=128,
        emb_keep_prob=0.8,
        grad_clip_norm=1,
        nobio_label_map_path=""
    )


# ------------
# Model Class
# ------------


class CharnerRNNModel(Model):
    """Charner RNN Model.
    Uses BiRNNs over sequence of input, and outputs i.i.d.
    predictions over the entire sequence.
    """

    def __init__(self, hparams):
        Model.__init__(self, hparams)

    def embed(self, *, inputs, params, is_training):
        """ Embedding byte or character level representations.
        Needs to be subclassed.
        """
        raise NotImplementedError

    def body(self, *, inputs, params, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # shape: [bs, padded_seq_len, emb_dim]
        embeddings = self.embed(inputs=inputs, is_training=is_training)

        # shape: [bs]
        seq_lengths = tf.reshape(
            inputs[Features.INPUT_SEQUENCE_LENGTH.value],
            [-1]
        )
        rnn_keep_prob = self.hp.dropout_keep_prob \
            if is_training \
            else [1. for _ in range(len(self.hp.dropout_keep_prob))]

        def cell_fn(drop):
            cell = tf.nn.rnn_cell.LSTMCell(self.hp.birnn_dim)
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=drop)

        fw_cells = [cell_fn(rnn_keep_prob[i]) for i in range(self.hp.birnn_layers)]
        bw_cells = [cell_fn(rnn_keep_prob[i]) for i in range(self.hp.birnn_layers)]

        # shape: [bs, padded_seq_len, rnn_hdim x 2]
        rnn_out, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw_cells,
            bw_cells,
            embeddings,
            sequence_length=seq_lengths,
            dtype=tf.float32
        )

        # shape: [bs, padded_seq_len, nlabels]
        logits = tf.layers.dense(rnn_out, self.hp.output_dim, use_bias=True)

        return logits

    def loss(self, *, predictions, features, targets, is_training):
        """ cross-entropy over sequence of tag predictions
        shapes:
            predictions: [bs, padded_seq_len, 5]
            features.seq_lens: [bs]
            targets: [bs]
        """

        # shape: [bs, padded_seq_len]
        seq_lengths = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value],
            [-1]
        )
        weights = tf.sequence_mask(seq_lengths, dtype=tf.float32)
        return tf.contrib.seq2seq.sequence_loss(predictions, targets, weights)

    def get_model_fn(self, model_dir=None):
        def fn(features, labels, mode, params):

            # shape: [bs, padded_seq_len, nlabels]
            logits = self.body(inputs=features, params=params, mode=mode)

            if mode == tf.estimator.ModeKeys.TRAIN:
                loss = self.loss(
                    predictions=logits,
                    features=features,
                    targets=labels,
                    is_training=True
                )
                train_op = get_training_op(loss, self.hp)
                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    train_op=train_op
                )

            if mode == tf.estimator.ModeKeys.EVAL:
                loss = self.loss(
                    predictions=logits,
                    features=features,
                    targets=labels,
                    is_training=False
                )

                predictions = tf.math.argmax(input=logits, axis=-1)
                seq_lengths = tf.reshape(
                    features[Features.INPUT_SEQUENCE_LENGTH.value],
                    [-1]
                )
                weights = tf.sequence_mask(seq_lengths, dtype=tf.float32)

                eval_metrics = {
                    'accuracy': tf.metrics.accuracy(labels,
                                                    predictions,
                                                    weights=weights)
                }

                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metric_ops=eval_metrics
                )

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    Predictions.RAW_TOKEN_SCORES.value: tf.nn.softmax(logits),
                    Predictions.LENGTH.value: features[Features.INPUT_SEQUENCE_LENGTH.value],
                    Predictions.WINDOW_ID.value: features[Features.WINDOW_ID.value]
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        return fn


@Registries.models.register("charner")
class CharnerBytes(CharnerRNNModel):
    """ CharNER model that operates over "raw" sequences of bytes. """

    def __init__(self, hparams):
        CharnerRNNModel.__init__(self, hparams)
        self.embed_shape = [256, self.hp.emb_dim]

    def embed(self, *, inputs, params, is_training):
        """ Byte representations with a randomly initialized embedding matrix.
        """

        keep_prob = self.hp.emb_keep_prob if is_training else 1.
        byte_inp = inputs[Features.INPUT_SYMBOLS.value]
        embedding = tf.get_variable("byte_embed_mat", shape=self.embed_shape)
        byte_inp = tf.nn.embedding_lookup(embedding, byte_inp)
        s = tf.shape(byte_inp)
        byte_inp = tf.nn.dropout(
            byte_inp,
            keep_prob=keep_prob,
            noise_shape=[s[0], s[1], 1]
        )
        return byte_inp
