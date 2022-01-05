"""
This file contains hotfix functions which will change the keras optimizer / 
model classes to enable pickling of these objects.

See also: https://github.com/tensorflow/tensorflow/issues/34697
"""

import tensorflow as tf
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from sources.flwr_parameters.set_random_seeds import set_seeds


def unpack_optimizer(optimizer_config, optimizer_weights):
    # Workaround for Bug (from_config
    # expects "lr" as key, but get_config gives "learning_rate")
    optimizer = tf.keras.optimizers.deserialize(optimizer_config)
    optimizer.set_weights(optimizer_weights)
    return optimizer


def make_keras_optimizers_pickleable():
    def __reduce__(self):
        optimizer_weights = self.get_weights()
        optimizer_config = tf.keras.optimizers.serialize(self)
        return unpack_optimizer, (optimizer_config, optimizer_weights)

    cls = tf.keras.optimizers.Optimizer
    cls.__reduce__ = __reduce__


def unpack_model(model, training_config, weights):
    set_seeds()
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_models_pickleable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return unpack_model, (model, training_config, weights)

    cls = tf.keras.models.Model
    cls.__reduce__ = __reduce__


make_keras_pickleable_executed = False


def make_keras_pickleable():
    global make_keras_pickleable_executed
    if not make_keras_pickleable_executed:
        make_keras_models_pickleable()
        make_keras_optimizers_pickleable()
        make_keras_pickleable_executed = True
