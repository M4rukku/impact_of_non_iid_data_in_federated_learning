import logging
import traceback

import flwr as fl
import numpy as np

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import \
    ClientDatasetFactory
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.utils.simulation_parameters import SimulationParameters
from sources.models.keras_model_template import KerasModelTemplate
from sources.simulators.base_client_provider import BaseClientProvider
from sources.simulators.base_simulator import BaseSimulator
from sources.simulators.serial_execution_simulator.start_serial_execution import \
    start_serial_simulation


class SerialExecutionSimulator(BaseSimulator):

    def __init__(self,
                 simulation_parameters: SimulationParameters,
                 strategy: fl.server.strategy.Strategy,
                 model_template: KerasModelTemplate,
                 dataset_factory: ClientDatasetFactory,
                 client_provider: BaseClientProvider = None,
                 seed: int = DEFAULT_SEED,
                 **kwargs
                 ):
        super().__init__(simulation_parameters,
                         strategy,
                         model_template,
                         dataset_factory,
                         client_provider,
                         seed,
                         **kwargs)

    def start_simulation(self):
        if "target_accuracy" in self.simulation_parameters and \
                self.simulation_parameters["target_accuracy"] is not None:
            logging.warning(
                "It appears like you wanted to use early stopping in combination with a target "
                "accuracy. Sadly, the SerialExecutionSimulator class does not yet "
                "support early stopping.")

        num_rounds = self.simulation_parameters["num_rounds"]
        num_clients = self.simulation_parameters["num_clients"]

        rng = np.random.default_rng(seed=self.seed)
        clients_ids = list(map(str, rng.choice(
            self.dataset_factory.get_number_of_clients(),
            size=num_clients,
            replace=False)))

        try:
            # Temporarily use a local copy of the ray based simulator
            start_serial_simulation(client_fn=self.client_provider,
                                    clients_ids=clients_ids,
                                    num_rounds=num_rounds,
                                    strategy=self.strategy)
        except Exception as e:
            logging.error(
                "Caught Exception after finishing Simulation - Caught the following error ")
            traceback.print_exception(type(e), e, e.__traceback__)
