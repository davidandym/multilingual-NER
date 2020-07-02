""" Abstract classes for building NER models. """


# pylint: disable=no-self-use,too-few-public-methods,invalid-name


from ner.registry import Registries


class Model():
    """ Abstract Model """

    def __init__(self, hparams):
        """ If `hparams` is a string it is looked up in the registry """
        if isinstance(hparams, str):
            hparams = Registries.hparams[hparams]

        self._hparams = hparams

    def body(self, *, inputs, params, mode):
        """ Most of the computation happens here """
        raise NotImplementedError

    def loss(self, *, predictions, features, targets, is_training):
        """ Compute loss given predictions and targets """
        raise NotImplementedError

    def get_model_fn(self, model_dir=None):
        """ Return a model function to be used by an `Estimator` """
        raise NotImplementedError

    @property
    def hp(self):
        return self._hparams


class Embedding():
    """ Embedding layer, that embeds raw input tokens """
    def __init__(self, hparams):
        self._params = hparams

    def embed(self, *, inputs, params, is_training):
        """ Embed function. Should return a vector representation of some form
        for each token in the input.
        """
        raise NotImplementedError

    @property
    def hp(self):
        return self._params
