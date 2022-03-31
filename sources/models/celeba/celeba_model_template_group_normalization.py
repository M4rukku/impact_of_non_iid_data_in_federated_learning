import tensorflow as tf
import tensorflow_addons as tfa
from sources.global_data_properties import CELEBA_IMAGE_SIZE
from sources.models.celeba.celeba_model_template import CelebaKerasModelTemplate


class CelebaModelTemplateGroupNormalization(CelebaKerasModelTemplate):

    def __init__(self, seed, group_size: int = 2):
        super().__init__(seed)
        self.group_size = group_size

    def get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(CELEBA_IMAGE_SIZE, CELEBA_IMAGE_SIZE, 3),
                                             dtype=tf.float32))
        for _ in range(4):
            model.add(tf.keras.layers.Conv2D(32, 3, padding='same'))

            channels = model.output.shape[-1]
            if channels % self.group_size != 0:
                raise RuntimeError(f"Error when applying group normalization with group size "
                                   f"{self.group_size}. The model has {channels} channels, "
                                   f"which is not divisible by the group size. (Requirement)")

            model.add(tfa.layers.GroupNormalization(groups=int(channels / self.group_size)))
            model.add(tf.keras.layers.MaxPooling2D(2, 2, padding='same'))
            model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.num_classes))
        model.add(tf.keras.layers.Softmax())

        return model
