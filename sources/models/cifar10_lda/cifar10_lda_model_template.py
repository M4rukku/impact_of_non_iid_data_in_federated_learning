"""
Code is adapted from flwr:
https://github.com/adap/flower/blob/e6aad8e4017d4efdc12af07d2f0b36b1f80cdc5d/src/py/flwr_example/pytorch_cifar/cifar.py

PyTorch CIFAR-10 image classification.
The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import tensorflow as tf
from typing import List, Union, Optional

from sources.global_data_properties import CIFAR_10_CLASSES, CIFAR_10_IMAGE_SIZE, \
    CIFAR_10_IMAGE_DIMENSIONS
from sources.metrics.default_metrics import get_default_sparse_categorical_metrics
from sources.models.model_template import ModelTemplate


class Cifar10LdaModelTemplate(ModelTemplate):

    def __init__(self, seed, num_classes=CIFAR_10_CLASSES,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):

        super(Cifar10LdaModelTemplate, self).__init__(seed, loss, num_classes)

    def get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(CIFAR_10_IMAGE_SIZE,
                                                          CIFAR_10_IMAGE_SIZE,
                                                          CIFAR_10_IMAGE_DIMENSIONS),
                                             dtype=tf.float32))

        model.add(tf.keras.layers.Conv2D(6, 5, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2, padding='same'))
        model.add(tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120, activation='relu'))
        model.add(tf.keras.layers.Dense(84, activation='relu'))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Softmax())
        return model

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return get_default_sparse_categorical_metrics(self.num_classes)

    def get_optimizer(self, lr=0.01, momentum=0.9, model: Optional[tf.keras.models.Model] = None) \
            -> tf.keras.optimizers.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
