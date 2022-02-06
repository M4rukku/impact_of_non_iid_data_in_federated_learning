from typing import List, Union, Optional
import tensorflow as tf

from sources.global_data_properties import CELEBA_IMAGE_SIZE, CELEBA_CLASSES
from sources.metrics.default_metrics import get_default_sparse_categorical_metrics
from sources.models.model_template import ModelTemplate


class CelebaModelTemplate(ModelTemplate):

    def __init__(self, seed, num_classes=CELEBA_CLASSES,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):

        super(CelebaModelTemplate, self).__init__(seed, loss, num_classes)

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

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.num_classes))
        model.add(tf.keras.layers.Softmax())

        return model

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return get_default_sparse_categorical_metrics(self.num_classes)

    def get_optimizer(self, lr=0.001, model: Optional[tf.keras.models.Model] = None) -> tf.keras.optimizers.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)
