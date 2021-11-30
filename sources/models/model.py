"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Union


class Model(ABC):
    def __init__(self, seed):
        self.seed = seed

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def get_default_optimizer(self, lr: Union[None, float] = None) -> tf.keras.optimizers.Optimizer:
        pass

    @abstractmethod
    def get_default_loss_function(self) -> tf.keras.losses.Loss:
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass
