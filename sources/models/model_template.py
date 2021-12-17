"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import tensorflow as tf
from sources.flwr_parameters.set_random_seeds import set_seeds


class ModelTemplate(ABC):
    def __init__(self, seed, optimizer, loss):
        self.seed = seed
        self.optimizer: tf.keras.optimizers.Optimizer = optimizer
        self.loss: tf.keras.losses.Loss = loss
        set_seeds(self.seed)

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        pass

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self.optimizer

    def get_loss(self) -> tf.keras.losses.Loss:
        return self.loss

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer):
        self.optimizer = optimizer
