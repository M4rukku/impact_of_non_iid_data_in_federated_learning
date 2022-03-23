import logging
import traceback

import flwr as fl
import numpy as np
import tensorflow as tf

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import \
    ClientDatasetFactory
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.flwr_parameters.simulation_parameters import SimulationParameters
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate
from sources.simulation_framework.simulators.base_simulator import BaseSimulator
from sources.simulation_framework.simulators.pickleable_client_function import PickleableClientFunction
from sources.simulation_framework.simulators.serial_execution_simulator.start_serial_execution import start_serial_simulation


class SerialExecutionSimulator(BaseSimulator):

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: ModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
                 evaluation_callbacks: list[tf.keras.callbacks.Callback] = None,
                 metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
                 seed: int = DEFAULT_SEED,
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

    def start_simulation(self):
        client_fn = PickleableClientFunction(self.model_template,
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
            start_serial_simulation(client_fn=client_fn,
                                    clients_ids=clients_ids,
                                    num_rounds=num_rounds,
                                    strategy=self.strategy)
        except Exception as e:
            logging.error(
                "Caught Exception after finishing Simulation - Caught the following error ")
            traceback.print_exception(type(e), e, e.__traceback__)
