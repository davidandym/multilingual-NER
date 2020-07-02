#! /usr/bin/env python
""" Main experiment script. """


# pylint: disable=too-many-locals,too-many-statements,invalid-name


import argparse as ap
import logging

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

import ner.data.dataset as ner_data
from ner.registry import Registries
from ner.hparams import (
    hparams_maybe_override,
    HParams,
    pretty_print_hparams
)


logging.getLogger().setLevel(logging.INFO)


def get_arguments():
    """ All cli arguments. """
    p = ap.ArgumentParser()
    p.add_argument('mode', type=str, choices=['train', 'predict'])

    # files
    p.add_argument('--train-file', type=str)
    p.add_argument('--dev-file', type=str)
    p.add_argument('--test-file', type=str)
    p.add_argument('--model-path', type=str, default='checkpoints')
    p.add_argument('--output-file', type=str, default='tags.txt')
    p.add_argument('--delete-existing', default=True)

    # Model hyperparams
    p.add_argument('--model', default='standard_word_crf', type=str)
    p.add_argument('--hparam-defaults', default='bi_crf_default_9', type=str)
    p.add_argument('--hparams-str', type=str,
                   help=("Update `hparams` from comma separated list of ",
                         "name=value pairs"))
    p.add_argument('--hparams-json', type=str,
                   help="Update `hparams` from parameters in JSON file")

    # training hyperparams
    p.add_argument('--save-checkpoints-steps', default=100)
    p.add_argument('--min-epochs-before-early-stop', type=int, default=5)
    p.add_argument('--early-stop-patience', type=int, default=10)
    p.add_argument('--train-epochs', default=1, type=int)
    p.add_argument('--shuffle-buffer-size', type=int, default=10000)
    p.add_argument('--random-seed', type=int, default=42)

    return p.parse_args()


# --------------------------------
# Dataset funcs
# --------------------------------


def train_dataset(params: HParams, iterator: ner_data.Generator):
    """ train function for tf estimator """
    data = Dataset.from_generator(
        iterator.generator(),
        iterator.datatypes(),
        iterator.datashape())

    data = data.shuffle(params.shuffle_buffer_size)
    data = data.padded_batch(params.batch_size, iterator.datashape())
    data = data.prefetch(None)
    return data


def eval_dataset(params: HParams, iterator: ner_data.Generator):
    """ test function for tf estimator """
    data = Dataset.from_generator(
        iterator.generator(),
        iterator.datatypes(),
        iterator.datashape())

    data = data.padded_batch(params.batch_size, iterator.datashape())
    return data


# --------------------------------
# Modes
# --------------------------------


def fit(args):
    """ Train a model. """
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    hparams = hparams_maybe_override(
        args.hparam_defaults,
        hparams_str=args.hparams_str,
        hparams_json=args.hparams_json)
    pretty_print_hparams(hparams)


    train_ds = ner_data.Dataset(args.train_file)
    dev_ds = ner_data.Dataset(args.dev_file)

    train_gen = Registries.train_generators[args.model](dataset=train_ds,
                                                        hp=hparams)
    eval_gen = Registries.test_generators[args.model](dataset=dev_ds,
                                                      hp=hparams)

    model = Registries.models[args.model](hparams)

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=args.save_checkpoints_steps,
        save_checkpoints_secs=None,
        keep_checkpoint_max=3,
        log_step_count_steps=100,
        save_summary_steps=100
    )

    estimator_params = train_gen.estimator_params()

    min_loss = np.PINF
    patience = 0

    for epoch in range(args.train_epochs):
        tf.logging.info('Starting epoch {}'.format(epoch))

        estimator = tf.estimator.Estimator(
            model_fn=model.get_model_fn(),
            model_dir=args.model_path,
            config=config,
            params=estimator_params)

        def train_input_fn():
            return train_dataset(hparams, train_gen)

        def eval_input_fn():
            return eval_dataset(hparams, eval_gen)

        estimator.train(input_fn=train_input_fn)

        tf.logging.info("** Evaluating epoch {} **".format(epoch))
        metrics = estimator.evaluate(input_fn=eval_input_fn)

        if metrics['loss'] < min_loss:
            min_loss = metrics['loss']
            patience = 0
        elif epoch > args.min_epochs_before_early_stop:
            patience += 1

        if patience >= args.early_stop_patience:
            tf.logging.info('Early stopping at epoch {}:\n\
                   No decrease in loss for {} epochs.\n\
                   Min loss achieved: {}'.format(epoch, patience, min_loss))
            break


def predict(args):
    """ Predict with model. """
    hparams = hparams_maybe_override(
        args.hparam_defaults,
        hparams_str=args.hparams_str,
        hparams_json=args.hparams_json)
    pretty_print_hparams(hparams)

    test_ds = ner_data.Dataset(args.test_file)
    test_gen = Registries.test_generators[args.model](dataset=test_ds,
                                                      hp=hparams)
    estimator_params = test_gen.estimator_params()

    model = Registries.models[args.model](hparams)

    estimator = tf.estimator.Estimator(
        model_fn=model.get_model_fn(),
        model_dir=args.model_path,
        params=estimator_params
    )

    def eval_input_fn():
        return eval_dataset(hparams, test_gen)

    predictions = estimator.predict(eval_input_fn)
    predictor = Registries.predictors[args.model](test_ds, hparams)
    predictor.gather(predictions)
    predictor.output_predictions(args.output_file)


if __name__ == '__main__':
    ARGS = get_arguments()
    MODE = ARGS.mode.lower()
    if MODE == 'train':
        fit(ARGS)
    elif MODE == 'predict':
        predict(ARGS)
    else:
        raise ValueError(MODE)
