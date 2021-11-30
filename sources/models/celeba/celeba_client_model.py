from typing import Union
import tensorflow as tf

from global_parameters import CELEBA_IMAGE_SIZE, CELEBA_CLASSES
from sources.models.model import Model


class CelebaClientModel(Model):

    def __init__(self, seed, num_classes=CELEBA_CLASSES):
        self.num_classes = num_classes
        super(CelebaClientModel, self).__init__(seed)

    def get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(CELEBA_IMAGE_SIZE, CELEBA_IMAGE_SIZE, 3),
                                             dtype=tf.float32))
        for _ in range(4):
            model.add(tf.keras.layers.Conv2D(32, 3, padding='same'))
            # Trainable usually says whether a layer should modify its arguments during training.
            # However, in the case of the BatchNormalization layer, setting trainable = False on the layer means that
            # the layer will be subsequently run in inference mode (meaning that it will use the moving mean and the
            # moving variance to normalize the current batch, rather than using the mean and variance of the current
            # batch).
            model.add(tf.keras.layers.BatchNormalization(trainable=True))
            model.add(tf.keras.layers.MaxPooling2D(2, 2, padding='same'))
            model.add(tf.keras.layers.ReLU())

        print(model.layers[-1].output_shape)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.num_classes))
        model.add(tf.keras.layers.Softmax())

        return model

    def get_default_optimizer(self, lr: Union[None, float] = None) -> tf.keras.optimizers.Optimizer:
        if lr is not None:
            return tf.keras.optimizers.SGD(lr=lr)
        else:
            return tf.keras.optimizers.SGD()

    def get_default_loss_function(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.SparseCategoricalCrossentropy()