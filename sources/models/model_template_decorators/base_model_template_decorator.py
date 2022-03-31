from typing import List, Union, Optional

import tensorflow as tf

from sources.models.model_template import ModelTemplate


class BaseModelTemplateDecorator(ModelTemplate):
    def __init__(self, model_template: ModelTemplate):
        super().__init__(model_template.seed, model_template.loss, model_template.num_classes)
        self.decorated_model_template = model_template

    def get_model(self) -> tf.keras.Model:
        return self.decorated_model_template.get_model()

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return self.decorated_model_template.get_centralised_metrics()

    def get_optimizer(self, lr=0.1, model: Optional[tf.keras.models.Model] = None) \
            -> tf.keras.optimizers.Optimizer:
        return self.decorated_model_template.get_optimizer(lr, model)

    def get_loss(self, model: Optional[tf.keras.models.Model] = None) -> tf.keras.losses.Loss:
        return self.decorated_model_template.get_loss(model)

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer):
        self.decorated_model_template.set_optimizer(optimizer)
