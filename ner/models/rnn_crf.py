""" Recurrent neural network model with CRF output layer """


# pylint: disable=no-name-in-module,no-self-use,assignment-from-no-return
# pylint: disable=inconsistent-return-statements,invalid-name
# pylint: disable=duplicate-code
# pylint: disable=unused-import


import tensorflow as tf
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn as BiRNN

from ner.registry import Registries
from ner.hparams import HParams
from ner.features import Features
from ner.features import Predictions
from ner.models.model import Model
from ner.models.word_encoder import BasicWordEncoder
from ner.optim import get_training_op


# ------------------------
# Hyperparameter Defaults
# ------------------------


# Byte-leve CRF Model
@Registries.hparams.register
def byte_level_crf():
    return HParams(
        vocab_file="",
        subword_file="",
        pretrained_embeddings=False,
        batch_size=128,
        shuffle_buffer_size=40000,
        char_pad_len=64,
        subword_pad_len=64,
        byte_pad_length=64,
        byte_enc_layers=2,
        byte_emb_dim=256,
        byte_enc_hdim=256,
        byte_enc_keep_prob=0.8,
        emb_enc_type_cnn=False,
        drop_single_rate=0.5,
        features="bytes",
        output_dim=0,
        birnn_layers=1,
        hidden_size=512,
        dropout_keep_prob=0.8,
        learning_rate=0.0001,
        optimizer='adam',
        beta1=0.9,
        beta2=0.999,
        use_ema=False,
        learning_rate_schedule='cosine_decay',
        first_decay_steps=1000,
        word_voc_size=0,
        char_voc_size=0,
        subword_voc_size=0
    )


# Byte-level CRF Model with word-embeddings
@Registries.hparams.register
def word_and_byte_crf():
    return HParams(
        vocab_file="",
        subword_file="",
        pretrained_embeddings=True,
        batch_size=128,
        shuffle_buffer_size=40000,
        char_pad_len=64,
        subword_pad_len=64,
        byte_pad_length=64,
        byte_enc_layers=2,
        byte_emb_dim=256,
        byte_enc_hdim=256,
        byte_enc_keep_prob=0.8,
        emb_enc_type_cnn=False,
        word_emb_dim=300,
        drop_single_rate=0.5,
        features='words+bytes',
        output_dim=0,
        birnn_layers=1,
        hidden_size=512,
        dropout_keep_prob=0.8,
        learning_rate=0.0001,
        optimizer='adam',
        beta1=0.9,
        beta2=0.999,
        use_ema=False,
        learning_rate_schedule='cosine_decay',
        first_decay_steps=1000,
        word_voc_size=0,
        char_voc_size=0,
        subword_voc_size=0
    )


# ------------
# Model Class
# ------------


class BiCRFModel(Model):
    """ Standard CRF model, using BiRNNs to encode at the token level.
    Override embed method to use special embeddings, i.e. BERT features.
    """

    def __init__(self, hparams):
        Model.__init__(self, hparams)
        self.scaffold = None
        self.transition_params = None

    def embed(self, *, inputs, is_training):
        """ BiCRF model requires an embed function to create token-level
        representations. Could come from pre-trained LMs, word-vectors, or
        learned sub-word encoders.
        """
        raise NotImplementedError

    def body(self, *, inputs, params, mode):
        """ Return token-level logits """

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Get token embeddings
        token_embeddings = self.embed(inputs=inputs, params=params, is_training=is_training)
        self.transition_params = tf.get_variable("crf-trans-params",
                                                 shape=[self.hp.output_dim,
                                                        self.hp.output_dim])

        # Build up sentence encoder
        rnn_keep_prob = self.hp.dropout_keep_prob if is_training else 1.

        def build_lstm_cell():
            cell = tf.nn.rnn_cell.LSTMCell(self.hp.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=rnn_keep_prob,
                input_keep_prob=rnn_keep_prob)
            return cell

        fw_cells = [build_lstm_cell() for _ in range(self.hp.birnn_layers)]
        bw_cells = [build_lstm_cell() for _ in range(self.hp.birnn_layers)]

        # encode inputs using RNN encoder
        seq_length = tf.reshape(
            inputs[Features.INPUT_SEQUENCE_LENGTH.value],
            [-1]
        )
        outputs, _, _ = BiRNN(fw_cells, bw_cells, token_embeddings,
                              sequence_length=seq_length, dtype=tf.float32)

        # convert encoded inputs to logits
        logits = tf.layers.dense(
            outputs,
            self.hp.output_dim,
            use_bias=True)

        return logits

    def loss(self, *, predictions, features, targets, is_training):
        """ For a CRF, predictions should be token-level logits and
        targets should be indexed labels.
        """
        del is_training
        seq_lens = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
        likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            predictions,
            targets,
            seq_lens,
            transition_params=self.transition_params)

        return tf.reduce_mean(-likelihood)

    def predict_from_logits(self, *, logits, features):
        """ Do CRF decoding starting from logits, rather than raw input """
        seq_lens = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value],
            [-1]
        )
        predictions, _ = tf.contrib.crf.crf_decode(
            logits,
            self.transition_params,
            seq_lens)
        return {
            Predictions.TAGS.value: predictions,
        }

    def get_model_fn(self, model_dir=None):
        def fn(features, labels, mode, params):

            is_training = mode == tf.estimator.ModeKeys.TRAIN
            logits = self.body(inputs=features, params=params, mode=mode)

            if is_training:
                loss = self.loss(predictions=logits,
                                 features=features,
                                 targets=labels,
                                 is_training=is_training)
                train_op = get_training_op(loss, self.hp)
                return tf.estimator.EstimatorSpec(mode,
                                                  loss=loss,
                                                  train_op=train_op,
                                                  scaffold=self.scaffold)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = self.predict_from_logits(
                    logits=logits,
                    features=features
                )

                # gradient extraction stuff - removing, as it's not a part of
                # the actual models used. This was used to help compute Fisher
                # information matrices.
                #
                # if 'return_logits' in self.hp:
                #     predictions['output-logits'] = logits
                #     predictions['output-tparams'] = tf.expand_dims(
                #         tf.identity(self.transition_params), 0
                #     )
                # if 'return_loss_grad' in self.hp:
                #     tf.logging.info(features)
                #     l = features['target-labels']

                #     seq_lens = tf.reshape(
                #             features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
                #     # transition_params = _get_transition_params(self.hp.output_dim)
                #     likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                #         logits,
                #         l,
                #         seq_lens,
                #         transition_params=self.transition_params)

                #     gradients = tf.gradients(likelihood, tf.trainable_variables())

                #     tf.logging.info(gradients)
                #     for grads in gradients:
                #         if isinstance(grads, tf.IndexedSlices):
                #             predictions[grads.name+"indces"] = tf.expand_dims(grads.indices, 0)
                #             predictions[grads.name+"values"] = tf.expand_dims(grads.values, 0)
                #         else:
                #             predictions[grads.name] = tf.expand_dims(grads, 0)

                predictions[Predictions.LENGTH.value] = tf.reshape(
                    features[Features.INPUT_SEQUENCE_LENGTH.value], [-1]
                )
                est = tf.estimator.EstimatorSpec(mode, predictions=predictions)
                return est

            if mode == tf.estimator.ModeKeys.EVAL:
                loss = self.loss(predictions=logits, features=features,
                                 targets=labels, is_training=is_training)

                predictions = self.predict_from_logits(
                    logits=logits, features=features)

                seq_lens = tf.reshape(features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
                weights = tf.sequence_mask(seq_lens, dtype=tf.float32)

                predicted_labels = predictions[Predictions.TAGS.value]

                eval_metrics = {
                    'accuracy': tf.metrics.accuracy(labels, predicted_labels, weights=weights)
                }

                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metric_ops=eval_metrics)

        return fn


@Registries.models.register('standard_word_crf')
class StandardWordCRF(BasicWordEncoder, BiCRFModel):
    """ Standard word-crf. Embedding layer supports encoding words via byte,
    character, or subword representations, in addition to word-embeddings.
    """
    def __init__(self, hparams):
        BiCRFModel.__init__(self, hparams)
        BasicWordEncoder.__init__(self, hparams)
