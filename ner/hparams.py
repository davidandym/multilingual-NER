""" Utilities for handling model hyper-parameters. """


# pylint: disable=invalid-name


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace
import tensorflow as tf

from ner.registry import Registries


HParams = tf.contrib.training.HParams


# -------------------------
# Hyperparameter Utilities
# -------------------------


def hparams_with_defaults(hparams, **defaults):
    """ Return HParams object with defaults """
    default_hparams = HParams(**defaults)
    return update_hparams(default_hparams, hparams)


def update_hparams(hparams, new_hparams):
    """ Update existing with new hyperparameters """
    if new_hparams is None:
        return hparams

    if isinstance(new_hparams, str) and new_hparams.endswith('.json'):
        tf.logging.info("Overriding default hparams from JSON")
        with open(new_hparams) as fh:
            hparams.parse_json(fh.read())
    elif isinstance(new_hparams, str):
        tf.logging.info("Overriding default hparams from str:")
        hparams.parse(new_hparams)
    elif isinstance(new_hparams, dict):
        tf.logging.info("Overriding default hparams from dict:")
        for k, val in new_hparams.items():
            if k in hparams:
                tf.logging.info("  {} -> {}".format(k, val))
                hparams.set_hparam(k, val)
    elif isinstance(new_hparams, Namespace):
        tf.logging.info("Overriding default hparams from Namespace:")
        for k, val in vars(new_hparams).items():
            if k in hparams and val is not None:
                tf.logging.info("  {} -> {}".format(k, val))
                hparams.set_hparam(k, val)
    else:
        raise ValueError(new_hparams)

    return hparams


def hparams_maybe_override(hparams, hparams_str=None, hparams_json=None):
    hparams = Registries.hparams[hparams]
    update_hparams(hparams, hparams_str)
    update_hparams(hparams, hparams_json)
    return hparams


def pretty_print_hparams(hparams):
    tf.logging.info("HParams:")
    for k, v in hparams.values().items():
        if isinstance(v, HParams):
            v = "<HParams Instance>"
        if isinstance(v, list):
            continue
        tf.logging.info(f"{k:30} {v:>25}")
