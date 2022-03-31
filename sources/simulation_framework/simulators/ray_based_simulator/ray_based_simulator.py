import logging
import traceback
from typing import Optional, List, Callable, TypedDict

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server import Server

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import ClientDatasetFactory
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.flwr_parameters.simulation_parameters import SimulationParameters, \
    RayInitArgs, ClientResources, DEFAULT_RAY_INIT_ARGS
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate
from sources.simulation_framework.simulators.base_simulator import BaseSimulator
from sources.simulation_framework.simulators.pickleable_base_client_provider import PickleableBaseClientProvider
from sources.simulation_framework.simulators.ray_based_simulator.ray_simulate import start_simulation


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
                 client_resources: ClientResources = {"num_gpus": 1, "num_cpus": 1},
                 ray_init_args: RayInitArgs = DEFAULT_RAY_INIT_ARGS,
                 ray_callbacks: Optional[List[Callable[[], None]]] = None,
                 server: Optional[Server] = None,
                 **kwargs
                 ):
        super().__init__(simulation_parameters,
                         strategy,
                         model_template,
                         dataset_factory,
                         fitting_callbacks,
                         evaluation_callbacks,
                         metrics,
                         seed,
                         **kwargs)

        self.client_resources = client_resources
        self.ray_init_args = ray_init_args
        self.server = server
        self.ray_callbacks = ray_callbacks

    def start_simulation(self):
        client_fn = PickleableBaseClientProvider(self.model_template,
                                                 self.dataset_factory,
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

        try:
            # Temporarily use a local copy of the ray based simulator
            start_simulation(client_fn=client_fn,
                             clients_ids=clients_ids,
                             client_resources=self.client_resources,
                             num_rounds=num_rounds,
                             strategy=self.strategy,
                             ray_init_args=self.ray_init_args,
                             server=self.server,
                             ray_callbacks=self.ray_callbacks)
        except Exception as e:
            logging.error(
                "Caught Exception after finishing Simulation - Caught the following error ")
            traceback.print_exception(type(e), e, e.__traceback__)


class DefaultRayArgumentDict(TypedDict):
    client_resources: ClientResources
    ray_init_args: RayInitArgs
    ray_callbacks: Optional[List[Callable[[], None]]]


default_ray_args: DefaultRayArgumentDict = {
    "ray_callbacks": None,
    "ray_init_args": DEFAULT_RAY_INIT_ARGS,
    "client_resources": {"num_gpus": 1, "num_cpus": 1}
}