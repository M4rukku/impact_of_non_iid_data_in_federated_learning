import tensorflow as tf
from typing import List


class ProximalLossDecorator(tf.keras.losses.Loss):
    def __init__(self,
                 keras_model_reference: tf.keras.models.Model,
                 decorated_loss: tf.keras.losses.Loss,
                 mu: tf.constant = tf.constant(1.0, dtype=tf.float32)
                 ):
        super().__init__(name=f"proximal_loss_{decorated_loss.name}")
        self.decorated_loss = decorated_loss
        self.keras_model_reference = keras_model_reference
        self.initial_weight_tensor = [tf.identity(weight)
                                      for weight in keras_model_reference.trainable_weights]
        self.mu = mu

    @staticmethod
    def calculate_norm_2_square(initial_tensor_list: List[tf.Tensor],
                                modified_tensor_list: List[tf.Tensor]):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                                 initial_tensor_list,
                                                 modified_tensor_list)
        squared_norm = tf.square(tf.linalg.global_norm(model_difference))
        return squared_norm

    def call(self, y_true, y_pred):
        orig_loss = self.decorated_loss.call(y_true, y_pred)
        proximal_term = (self.mu / 2) * self.calculate_norm_2_square(
            self.keras_model_reference.trainable_weights,
            self.initial_weight_tensor
        )
        fedprox_loss = orig_loss + proximal_term

        return fedprox_loss
