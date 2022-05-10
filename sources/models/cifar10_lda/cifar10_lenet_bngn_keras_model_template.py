import math

import tensorflow as tf
from tensorflow import keras
from typing import List, Union, Optional

from sources.global_data_properties import CIFAR_10_CLASSES, CIFAR_10_IMAGE_SIZE, \
    CIFAR_10_IMAGE_DIMENSIONS
from sources.metrics.default_metrics_tf import get_default_sparse_categorical_metrics_tf
from sources.models.cifar10_lda.bngn_lenet_model import get_lenet_model
from sources.models.keras_model_template import KerasModelTemplate


class Cifar10LdaLeNetKerasModelTemplate(KerasModelTemplate):

    def __init__(self, seed,
                 num_classes=CIFAR_10_CLASSES,
                 group_norm=True,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):
        self.group_norm = group_norm
        super(Cifar10LdaLeNetKerasModelTemplate, self).__init__(seed, loss, num_classes)

    def get_model(self) -> tf.keras.Model:
        inputs = keras.Input(shape=(CIFAR_10_IMAGE_SIZE,
                                    CIFAR_10_IMAGE_SIZE,
                                    CIFAR_10_IMAGE_DIMENSIONS),
                             dtype=tf.float32)

        model = get_lenet_model(inputs, self.group_norm)

        return model

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return get_default_sparse_categorical_metrics_tf(self.num_classes)

    def get_optimizer(self, lr=math.pow(10, -0.5), model: Optional[tf.keras.models.Model] = None) \
            -> tf.keras.optimizers.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)
