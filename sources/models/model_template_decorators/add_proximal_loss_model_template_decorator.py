from typing import Optional

import tensorflow as tf
from sources.models.loss_decorators.proximal_loss_decorator import ProximalLossDecorator
from sources.models.model_template import ModelTemplate
from sources.models.model_template_decorators.base_model_template_decorator import \
    BaseModelTemplateDecorator


class AddProximalLossModelTemplateDecorator(BaseModelTemplateDecorator):

    def __init__(self, decorated_model_template: ModelTemplate, mu: float = 0.2):
        super().__init__(decorated_model_template)
        self.mu = tf.constant(mu, dtype=tf.float32)

    def get_loss(self, model: Optional[tf.keras.models.Model] = None) -> tf.keras.losses.Loss:
        if model is None:
            raise RuntimeError("You need to pass the model into the get_loss function to use the "
                               "AddProximalLossModelTemplateDecorator. FedProx uses the weight "
                               "differences to calculate the proximal term and, therefore needs "
                               "access to the model.")
        return ProximalLossDecorator(model, self.decorated_model_template.get_loss(model), self.mu)
