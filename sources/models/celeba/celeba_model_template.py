import tensorflow as tf

from global_parameters import CELEBA_IMAGE_SIZE, CELEBA_CLASSES
from sources.models.model_template import ModelTemplate


class CelebaModelTemplate(ModelTemplate):

    def __init__(self, seed, num_classes=CELEBA_CLASSES,
                 optimizer=tf.keras.optimizers.SGD(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):

        self.num_classes = num_classes
        super(CelebaModelTemplate, self).__init__(seed, optimizer, loss)

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
