import flwr as fl
import numpy as np
import tensorflow as tf

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.flwr_clients.base_client import BaseClient
from sources.flwr_parameters.default_parameters import DEFAULT_SEED
from sources.flwr_parameters.simulation_parameters import SimulationParameters, \
    RayInitArgs, ClientResources, DEFAULT_RAY_INIT_ARGS
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate
from sources.simulation_framework.base_simulator import BaseSimulator


class RayBasedSimulator(BaseSimulator):

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: ModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
                 evaluation_callbacks: list[tf.keras.callbacks.Callback] = None,
                 metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
                 seed: int = DEFAULT_SEED,
                 client_resources: ClientResources = None,
                 ray_init_args: RayInitArgs = DEFAULT_RAY_INIT_ARGS
                 ):
        super().__init__(simulation_parameters,
                         strategy,
                         model_template,
                         dataset_factory,
                         fitting_callbacks,
                         evaluation_callbacks,
                         metrics,
                         seed)

        self.client_resources = client_resources
        self.ray_init_args = ray_init_args

    def start_simulation(self):
        def client_fn(client_identifier: str):
            return BaseClient(self.model_template,
                              self.dataset_factory.
                              create_dataset(client_identifier),
                              self.metrics,
                              self.fitting_callbacks,
                              self.evaluation_callbacks)

        num_rounds = self.simulation_parameters["num_rounds"]
        num_clients = self.simulation_parameters["num_clients"]

        rng = np.random.default_rng(seed=self.seed)
        clients_ids = list(map(str, rng.choice(
            self.dataset_factory.get_number_of_clients(),
            size=num_clients,
            replace=False)))

        fl.simulation.start_simulation(client_fn=client_fn,
                                       clients_ids=clients_ids,
                                       client_resources=self.client_resources,
                                       num_rounds=num_rounds,
                                       strategy=self.strategy,
                                       ray_init_args=self.ray_init_args)
