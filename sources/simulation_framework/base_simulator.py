import os
from abc import abstractmethod, ABC

import flwr as fl
import tensorflow as tf

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import ClientDatasetFactory
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.flwr_parameters.simulation_parameters import SimulationParameters
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.make_keras_pickleable import make_keras_pickleable
from sources.models.model_template import ModelTemplate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseSimulator(ABC):

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: ModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
                 evaluation_callbacks: list[tf.keras.callbacks.Callback] = None,
                 metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
                 seed: int = DEFAULT_SEED):
        make_keras_pickleable()

        self.simulation_parameters: SimulationParameters = simulation_parameters
        self.strategy: fl.server.strategy.Strategy = strategy
        self.model_template: ModelTemplate = model_template
        self.dataset_factory: ClientDatasetFactory = dataset_factory
        self.fitting_callbacks: list[
            tf.keras.callbacks.Callback] = fitting_callbacks
        self.evaluation_callbacks: list[
            tf.keras.callbacks.Callback] = evaluation_callbacks
        self.metrics: list[tf.keras.metrics.Metric] = metrics
        self.seed = seed

    @abstractmethod
    def start_simulation(self):
        pass


