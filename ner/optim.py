""" Optimization library. """


# pylint: disable=import-error,no-name-in-module,missing-docstring,invalid-name
# pylint: disable=too-few-public-methods


import tensorflow as tf
from ner.registry import Registries


@Registries.optimizers.register
def adam(learning_rate, hparams):
    return tf.train.AdamOptimizer(
        learning_rate,
        beta1=hparams.beta1,
        beta2=hparams.beta2)


@Registries.optimizers.register
def momentum(learning_rate, hparams):
    return tf.train.MomentumOptimizer(
        learning_rate,
        momentum=hparams.momentum)


@Registries.optimizers.register
def lars(learning_rate, hparams):
    return tf.contrib.opt.LARSOptimizer(
        learning_rate,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay,
        epsilon=hparams.epsilon)


@Registries.optimizers.register
def adam_w(learning_rate, hparams):
    return tf.contrib.opt.AdamWOptimizer(
        weight_decay=hparams.weight_decay,
        learning_rate=learning_rate,
        beta1=hparams.beta1,
        beta2=hparams.beta2
    )


@Registries.optimizers.register
def adafactor(learning_rate, hparams):
    try:
        from tensor2tensor.utils import adafactor as af
    except ImportError:
        print(
            ("Adafactor requires the tensor2tensor library."
             "Please run 'pip install tensor2tensor' to get Adafactor."))
        raise ImportError

    del learning_rate
    del hparams
    return af.AdafactorOptimizer()


def track_params_averages():
    """ Returns EMA object and average of params """
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    params = tf.trainable_variables()
    params_averages_op = ema.apply(params)
    return ema, params_averages_op


def get_training_op(loss, hparams):
    """ Returns op based on given hparams """
    lr = hparams.get("learning_rate", 1.0)
    if "learning_rate_schedule" in hparams:
        lr = Registries.learning_rates[hparams.learning_rate_schedule](
            learning_rate=lr, 
            hparams=hparams
        )
    opt = Registries.optimizers[hparams.optimizer](lr, hparams)

    if "grad_clip_norm" in hparams:
        opt = tf.contrib.estimator.clip_gradients_by_norm(
            opt, hparams.grad_clip_norm)

    var_list = None
    if "vars_to_opt" in hparams:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     hparams.vars_to_opt)
    train_op = opt.minimize(loss,
                            global_step=tf.train.get_global_step(),
                            var_list=var_list)
    return train_op
