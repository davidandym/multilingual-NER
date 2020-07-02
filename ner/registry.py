"""Object registration.

Registries are instances of `Registry`.

See `Registries` for a centralized list of object registries
(models, hyperparameter sets, etc.).

New functions and classes can be registered using `.register`. The can be
accessed/queried similar to dictionaries, keyed by default by `snake_case`
equivalents.

```
@Registries.models.register
class MyModel(Model):
  ...

'my_model' in Registries.models  # True
for k in Registries.models:
  print(k)  # prints 'my_model'
model = Registries.models['my_model'](constructor_arg)
```

See all the models registered: `list_models()`.

For hyperparameter sets:
  * Register: `register_hparams`
  * List: `list_hparams`
  * Retrieve by name: `hparams`
  * Command-line flag in `t2t_trainer.py`: `--hparams_set=name`

"""


# pylint: disable=no-name-in-module
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import logging
import re

import tensorflow as tf

from tensorflow.python.util import tf_inspect as inspect


_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def camelcase_to_snakecase(name):
    """ Convert CamelCase to snake_case """
    s1 = _first_cap_re.sub(r"\1_\2", name)
    return _all_cap_re.sub(r"\1_\2", s1).lower()


def snakecase_to_camelcase(name):
    """ Convert snake_case to CamelCase """
    return "".join([w[0].upper() + w[1:] for w in name.split("_")])


def pprint_hparams(hparams):
    """Represents hparams using its dictionary and calls pprint.pformat
    on it."""
    return "\n{}".format(pprint.pformat(hparams.values(), width=1))


def default_name(class_or_fn):
    """Default name for a class or function.

    This is the naming function by default for registries expecting classes or
    functions.

    Args:
        class_or_fn: class or function to be named.

    Returns:
        Default name for registration.
    """
    return camelcase_to_snakecase(class_or_fn.__name__)


def default_object_name(obj):
    """ Return default name for object """
    return default_name(type(obj))


class Registry():
    """Dict-like class for managing function registrations.

    ```python
    my_registry = Registry("custom_name")

    @my_registry.register
    def my_func():
        pass

    @my_registry.register()
    def another_func():
        pass

    @my_registry.register("non_default_name")
    def third_func(x, y, z):
        pass

    def foo():
        pass

    my_registry.register()(foo)
    my_registry.register("baz")(lambda (x, y): x + y)
    my_register.register("bar")

    print(list(my_registry))
    # ["my_func", "another_func", "non_default_name", "foo", "baz"]
    # (order may vary)
    print(my_registry["non_default_name"] is third_func)  # True
    print("third_func" in my_registry)                    # False
    print("bar" in my_registry)                           # False
    my_registry["non-existent_key"]                       # raises KeyError
    ```

    Optional validation, on_set callback and value transform also supported.
    See `__init__` doc.
    """

    def __init__(self,
                 registry_name,
                 default_key_fn=default_name,
                 validator=None,
                 on_set=None,
                 value_transformer=(lambda k, v: v)):
        """Construct a new registry.

         Args:
          registry_name: str identifier for the given registry.
            Used in error msgs.
          default_key_fn (optional): function mapping value -> key for
            registration when a key is not provided
          validator (optional): if given, this is run before setting a given
            pair. Accepts (key, value) and should raise if there is a
            problem. Overwriting existing keys is not allowed and is checked
            separately. Values are also checked to be callable separately.
          on_set (optional): callback function accepting (key, value) pair
            which is run after an item is successfully set.
          value_transformer (optional): if run, `__getitem__` will return
            value_transformer(key, registered_value).
        """
        self._registry = {}
        self._name = registry_name
        self._default_key_fn = default_key_fn
        self._validator = validator
        self._on_set = on_set
        self._value_transformer = value_transformer

    def default_key(self, value):
        """Default key used when key not provided. Uses function from __init__."""
        return self._default_key_fn(value)

    @property
    def name(self):
        """Return name"""
        return self._name

    def validate(self, key, value):
        """Validation function run before setting.
        Uses function from __init__."""
        if self._validator is not None:
            self._validator(key, value)

    def on_set(self, key, value):
        """Callback called on successful set. Uses function from __init__."""
        if self._on_set is not None:
            self._on_set(key, value)

    def __setitem__(self, key, value):
        """Validate, set, and (if successful) call `on_set` for the given item.

        Args:
          key: key to store value under. If `None`, `self.default_key(value)`
            is used.
          value: callable stored under the given key.

        Raises:
          KeyError: if key is already in registry.
        """
        if key is None:
            key = self.default_key(value)
        if key in self:
            raise KeyError(
                "key %s already registered in registry %s" % (key, self._name))
        if not callable(value):
            raise ValueError("value must be callable")
        self.validate(key, value)
        self._registry[key] = value
        self.on_set(key, value)

    def register(self, key_or_value=None):
        """Decorator to register a function, or registration itself.

        This is primarily intended for use as a decorator, either with
        or without a key/parentheses.

        ```python
        @my_registry.register('key1')
        def value_fn(x, y, z):
          pass

        @my_registry.register()
        def another_fn(x, y):
          pass

        @my_registry.register
        def third_func():
          pass
        ```

        Note if key_or_value is provided as a non-callable, registration only
        occurs once the returned callback is called with a callable as its only
        argument.
        ```python
        callback = my_registry.register('different_key')
        'different_key' in my_registry  # False
        callback(lambda (x, y): x + y)
        'different_key' in my_registry  # True
        ```

        Args: key_or_value (optional): key to access the registered
          value with, or the function itself. If `None` (default),
          `self.default_key` will be called on `value` once the
          returned callback is called with `value` as the only arg. If
          `key_or_value` is itself callable, it is assumed to be the
          value and the key is given by `self.default_key(key)`.

        Returns:
          decorated callback, or callback generated a decorated function.

        """

        def decorator(value, key):
            self[key] = value
            return value

        # Handle if decorator was used without parens
        if callable(key_or_value):
            return decorator(value=key_or_value, key=None)

        return lambda value: decorator(value, key=key_or_value)

    def __getitem__(self, key):
        if key not in self:
            tf.logging.error(
                "%s never registered with registry %s. Available:\n %s" %
                (key, self.name, display_list_by_prefix(sorted(self), 4)))
            raise KeyError(key)
        value = self._registry[key]
        return self._value_transformer(key, value)

    def __contains__(self, key):
        return key in self._registry

    def keys(self):
        return self._registry.keys()

    def values(self):
        return (self[k] for k in self)  # complicated because of transformer

    def items(self):
        return ((k, self[k]) for k in self)  # complicated because of transformer

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def _clear(self):
        self._registry.clear()

    def get(self, key, default=None):
        return self[key] if key in self else default


def _on_model_set(k, v):
    v.REGISTERED_NAME = k


def _nargs_validator(nargs, message):
    """Makes validator for function to ensure it takes nargs args."""
    if message is None:
        message = "Registered function must take exactly %d arguments" % nargs

    def f(key, value):
        del key
        spec = inspect.getfullargspec(value)
        if (len(spec.args) != nargs or spec.varargs is not None or spec.varkw is not None):
            raise ValueError(message)

    return f


def _hparams_value_transformer(key, value):
    out = value()
    if out is None:
        raise TypeError("HParams %s is None. Make sure the registered function "
                        "returns the HParams object" % key)
    return out


# pylint: disable=too-few-public-methods
class Registries():
    """Object holding `Registry` objects."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    models = Registry("models", on_set=_on_model_set)

    optimizers = Registry(
        "optimizers",
        validator=_nargs_validator(
            2,
            "Registered optimizer functions must take exactly two arguments: "
            "learning_rate (float) and hparams (HParams)."))

    learning_rates = Registry(
        "learning_rates",
        validator=_nargs_validator(
            2,
            "Registered learning rate function must take exactly two arguments:"
            " learning_rate (float) and hparams (HParams)."))

    hparams = Registry("hparams", value_transformer=_hparams_value_transformer)

    # model-specific generators and predictions
    train_generators = Registry('train_generators')
    test_generators = Registry('test_generators')
    predictors = Registry('predictors')

def optimizer(name):
    """Get pre-registered optimizer keyed by name.

    `name` should be snake case, though SGD -> sgd, RMSProp ->
    rms_prop and UpperCamelCase -> snake_case conversions included for
    legacy support.

    Args:
      name: name of optimizer used in registration. This should be a snake case
        identifier, though others supported for legacy reasons.

    Returns:
      optimizer

    """
    warn_msg = ("Please update `registry.optimizer` callsite "
                "(likely due to a `HParams.optimizer` value)")
    if name == "SGD":
        name = "sgd"
        logging.warning("'SGD' optimizer now keyed by 'sgd'. %s", warn_msg)
    elif name == "RMSProp":
        name = "rms_prop"
        logging.warning(
            "'RMSProp' optimizer now keyed by 'rms_prop'. %s", warn_msg)
    else:
        snake_name = camelcase_to_snakecase(name)
        if name != snake_name:
            logging.warning(
                "optimizer names now keyed by snake_case names. %s", warn_msg)
        name = snake_name

    return Registries.optimizers[name]


def display_list_by_prefix(names_list, starting_spaces=0):
    """Creates a help string for names_list grouped by prefix."""

    cur_prefix, result_lines = None, []
    space = " " * starting_spaces
    for name in sorted(names_list):
        split = name.split("_", 1)
        prefix = split[0]
        if cur_prefix != prefix:
            result_lines.append(space + prefix + ":")
            cur_prefix = prefix
        result_lines.append(space + "  * " + name)
    return "\n".join(result_lines)
