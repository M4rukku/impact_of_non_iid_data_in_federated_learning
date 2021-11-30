from typing import Union
import tensorflow as tf

from global_parameters import FEMNIST_IMAGE_SIZE, FEMNIST_CLASSES
from sources.models.model import Model


class FemnistClientModel(Model):

    def __init__(self, seed, num_classes=FEMNIST_CLASSES):
        self.num_classes = num_classes
        super(FemnistClientModel, self).__init__(seed)

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

    def get_default_optimizer(self, lr: Union[None, float] = None) -> tf.keras.optimizers.Optimizer:
        if lr is not None:
            return tf.keras.optimizers.SGD(lr=lr)
        else:
            return tf.keras.optimizers.SGD()

    def get_default_loss_function(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.SparseCategoricalCrossentropy()
