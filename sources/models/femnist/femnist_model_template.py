import tensorflow as tf
from typing import List, Union

from sources.global_data_properties import FEMNIST_IMAGE_SIZE, FEMNIST_CLASSES
from sources.metrics.default_metrics import get_default_sparse_categorical_metrics
from sources.models.model_template import ModelTemplate


class FemnistModelTemplate(ModelTemplate):

    def __init__(self, seed, num_classes=FEMNIST_CLASSES,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):

        super(FemnistModelTemplate, self).__init__(seed, loss, num_classes)

    def get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=([FEMNIST_IMAGE_SIZE, FEMNIST_IMAGE_SIZE, 1]),
                                             dtype=tf.float32))

        model.add(tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))

        model.add(tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.num_classes))
        model.add(tf.keras.layers.Softmax())

        return model

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return get_default_sparse_categorical_metrics(self.num_classes)

    def get_optimizer(self, lr=0.001) -> tf.keras.optimizers.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)