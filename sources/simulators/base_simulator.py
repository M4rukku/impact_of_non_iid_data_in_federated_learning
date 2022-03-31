import logging
import os
from abc import abstractmethod, ABC

import flwr as fl
import tensorflow as tf

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import \
    ClientDatasetFactory
from sources.models.base_model_template import BaseModelTemplate
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.utils.simulation_parameters import SimulationParameters
from sources.metrics.default_metrics_tf import DEFAULT_METRICS
from sources.simulators.base_client_provider import BaseClientProvider
from sources.simulators.keras_client_provider import KerasClientProvider

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseSimulator(ABC):

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: BaseModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 client_provider: BaseClientProvider,
                 metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
                 seed: int = DEFAULT_SEED,
                 **kwargs):
        self.simulation_parameters: SimulationParameters = simulation_parameters
        self.strategy: fl.server.strategy.Strategy = strategy
        self.model_template: BaseModelTemplate = model_template
        self.dataset_factory: ClientDatasetFactory = dataset_factory
        self.metrics: list[tf.keras.metrics.Metric] = metrics
        self.seed = seed

        if client_provider is None:
            logging.warning("BaseClientProvider passed to BaseSimulator is None - Setting it to "
                            "KerasClient by default. If this is not what you expected, "
                            "please specify the behaviour by passing the base_client_provider as "
                            "kwarg")
            # Ensuring Backwards Compatibility - Can be deleted once my project is done
            if "fitting_callbacks" in kwargs:
                fitting_callbacks = kwargs["fitting_callbacks"]
            else:
                fitting_callbacks = None
            if "evaluation_callbacks" in kwargs:
                evaluation_callbacks = kwargs["evaluation_callbacks"]
            else:
                evaluation_callbacks = None

            self.client_provider = KerasClientProvider(
                model_template=self.model_template,
                dataset_factory=self.dataset_factory,
                metrics=self.metrics,
                fitting_callbacks=fitting_callbacks,
                evaluation_callbacks=evaluation_callbacks
            )
        else:
            self.client_provider = client_provider

    @abstractmethod
    def start_simulation(self):
        pass
