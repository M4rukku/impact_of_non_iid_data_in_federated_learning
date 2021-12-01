import os
import time
from multiprocessing import Process

import flwr as fl
import numpy as np
import tensorflow as tf

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.flwr_clients.base_client import BaseClient
from sources.flwr_parameters.default_parameters import DEFAULT_SEED
from sources.flwr_parameters.simulation_parameters import SimulationParameters, DEFAULT_SERVER_ADDRESS
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def start_client(model_template, dataset_factory, client_identifier, metrics, fitting_callbacks, evaluation_callbacks):
    client = BaseClient(model_template, dataset_factory.create_dataset(str(client_identifier)),
                        metrics, fitting_callbacks, evaluation_callbacks)
    fl.client.start_numpy_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)


def start_server(strategy: fl.server.strategy.Strategy, simulation_parameters: SimulationParameters):
    fl.server.start_server(server_address=DEFAULT_SERVER_ADDRESS, strategy=strategy,
                           config={"num_rounds": simulation_parameters["num_rounds"]})


class BaseSimulator:

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: ModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
                 evaluation_callbacks: list[tf.keras.callbacks.Callback] = None,
                 metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
                 seed: int = DEFAULT_SEED):

        self.simulation_parameters: SimulationParameters = simulation_parameters
        self.strategy: fl.server.strategy.Strategy = strategy
        self.model_template: ModelTemplate = model_template
        self.dataset_factory: ClientDatasetFactory = dataset_factory
        self.fitting_callbacks: list[tf.keras.callbacks.Callback] = fitting_callbacks
        self.evaluation_callbacks: list[tf.keras.callbacks.Callback] = evaluation_callbacks
        self.metrics: list[tf.keras.metrics.Metric] = metrics
        self.seed = seed

    def start_simulation(self):
        processes = []

        # Start the server
        server_process = Process(
            target=start_server, args=(self.strategy, self.simulation_parameters), daemon=True
        )
        server_process.start()
        processes.append(server_process)

        # Optionally block the script here for a second or two so the server has time to start
        time.sleep(2)

        rng = np.random.default_rng(seed=self.seed)

        for client_identifier in rng.choice(self.dataset_factory.get_number_of_clients(),
                                            size=self.simulation_parameters["num_clients"],
                                            replace=False):
            client_process = Process(target=start_client, args=(self.model_template, self.dataset_factory,
                                                                client_identifier, self.metrics, self.fitting_callbacks,
                                                                self.evaluation_callbacks), daemon=True)
            client_process.start()
            processes.append(client_process)

        for p in processes:
            p.join()
