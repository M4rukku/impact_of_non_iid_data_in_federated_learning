"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
from typing import List, Union

import tensorflow as tf


class ModelTemplate(ABC):
    def __init__(self, seed, loss, num_classes=None):
        self.seed = seed
        self.num_classes = num_classes
        self.optimizer: tf.keras.optimizers.Optimizer = None
        self.loss: tf.keras.losses.Loss = loss
        # set_seeds(self.seed)

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        pass

    @abstractmethod
    def get_optimizer(self, lr=0.1) -> tf.keras.optimizers.Optimizer:
        pass

    def get_loss(self) -> tf.keras.losses.Loss:
        return self.loss

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer):
        self.optimizer = optimizer
