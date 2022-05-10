"""
Code is adapted from flwr:
https://github.com/adap/flower/blob/e6aad8e4017d4efdc12af07d2f0b36b1f80cdc5d/src/py/flwr_example/pytorch_cifar/cifar.py

PyTorch CIFAR-10 image classification.
The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import math

import tensorflow as tf
from tensorflow import keras
from typing import List, Union, Optional

from sources.global_data_properties import CIFAR_10_CLASSES, CIFAR_10_IMAGE_SIZE, \
    CIFAR_10_IMAGE_DIMENSIONS
from sources.metrics.default_metrics_tf import get_default_sparse_categorical_metrics_tf
from sources.models.cifar10_lda.resnet_18_batch_norm_model import resnet18_batch_norm_softmax
from sources.models.cifar10_lda.resnet_18_group_norm import resnet18_group_norm_softmax
from sources.models.keras_model_template import KerasModelTemplate


class Cifar10LdaResnet18KerasModelTemplate(KerasModelTemplate):

    def __init__(self, seed,
                 num_classes=CIFAR_10_CLASSES,
                 group_norm=True,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):
        self.group_norm = group_norm
        super(Cifar10LdaResnet18KerasModelTemplate, self).__init__(seed, loss, num_classes)

    def get_model(self) -> tf.keras.Model:
        inputs = keras.Input(shape=(CIFAR_10_IMAGE_SIZE,
                                    CIFAR_10_IMAGE_SIZE,
                                    CIFAR_10_IMAGE_DIMENSIONS),
                             dtype=tf.float32)

        if self.group_norm is True:
            outputs = resnet18_group_norm_softmax(inputs, self.num_classes)
        else:
            outputs = resnet18_batch_norm_softmax(inputs, self.num_classes)
        model = keras.Model(inputs, outputs)

        return model

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return get_default_sparse_categorical_metrics_tf(self.num_classes)

    def get_optimizer(self, lr=math.pow(10, -0.5), model: Optional[tf.keras.models.Model] = None) \
            -> tf.keras.optimizers.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)
