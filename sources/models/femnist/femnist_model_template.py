import tensorflow as tf

from sources.global_data_properties import FEMNIST_IMAGE_SIZE, FEMNIST_CLASSES
from sources.models.model_template import ModelTemplate


class FemnistModelTemplate(ModelTemplate):

    def __init__(self, seed, num_classes=FEMNIST_CLASSES,
                 optimizer=tf.keras.optimizers.SGD(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):
        self.num_classes = num_classes
        super(FemnistModelTemplate, self).__init__(seed, optimizer, loss)

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
