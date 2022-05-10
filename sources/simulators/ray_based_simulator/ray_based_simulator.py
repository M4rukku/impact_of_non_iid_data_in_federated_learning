import logging
import traceback
from typing import Optional, List, Callable, TypedDict

import flwr as fl
import numpy as np
from flwr.server import Server, SimpleClientManager

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import \
    ClientDatasetFactory
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.utils.simulation_parameters import SimulationParameters, \
    RayInitArgs, ClientResources, DEFAULT_RAY_INIT_ARGS, EarlyStoppingSimulationParameters
from sources.models.keras_model_template import KerasModelTemplate
from sources.flwr.flwr_servers.early_stopping_server import EarlyStoppingServer
from sources.simulators.base_client_provider import BaseClientProvider
from sources.simulators.base_simulator import BaseSimulator
from sources.simulators.ray_based_simulator.ray_simulate import \
    start_simulation


class RayBasedSimulator(BaseSimulator):

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: KerasModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 client_provider: BaseClientProvider = None,
                 seed: int = DEFAULT_SEED,
                 client_resources=None,
                 ray_init_args: RayInitArgs = DEFAULT_RAY_INIT_ARGS,
                 ray_callbacks: Optional[List[Callable[[], None]]] = None,
                 server: Optional[Server] = None,
                 **kwargs
                 ):
        super().__init__(simulation_parameters,
                         strategy,
                         model_template,
                         dataset_factory,
                         client_provider,
                         seed,
                         **kwargs)

        if client_resources is None:
            client_resources = {"num_gpus": 1, "num_cpus": 1}
        self.client_resources = client_resources
        self.ray_init_args = ray_init_args
        self.server = server
        self.ray_callbacks = ray_callbacks

        if "target_accuracy" in simulation_parameters and \
                simulation_parameters["target_accuracy"] is not None:
            simulation_parameters: EarlyStoppingSimulationParameters = simulation_parameters
            self.server = EarlyStoppingServer(SimpleClientManager(),
                                              strategy,
                                              simulation_parameters["target_accuracy"],
                                              simulation_parameters["num_rounds_above_target"]
                                              )

    def start_simulation(self):
        num_rounds = self.simulation_parameters["num_rounds"]
        num_clients = self.simulation_parameters["num_clients"]

        rng = np.random.default_rng(seed=self.seed)
        clients_ids = list(map(str, rng.choice(
            self.dataset_factory.get_number_of_clients(),
            size=num_clients,
            replace=False)))

        try:
            # Temporarily use a local copy of the ray based simulator
            start_simulation(client_fn=self.client_provider,
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


# noinspection PyTypeChecker
default_ray_args: DefaultRayArgumentDict = {
    "ray_callbacks": None,
    "ray_init_args": DEFAULT_RAY_INIT_ARGS,
    "client_resources": {"num_gpus": 1, "num_cpus": 1}
}
