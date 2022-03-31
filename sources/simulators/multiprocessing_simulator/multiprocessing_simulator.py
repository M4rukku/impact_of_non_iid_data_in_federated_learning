import time
from multiprocessing.context import Process

import numpy as np

from sources.simulators.base_simulator import BaseSimulator
from sources.simulators.multiprocessing_simulator.start_client import start_client
from sources.simulators.multiprocessing_simulator.start_server import start_server


class MultiprocessingBasedSimulator(BaseSimulator):
    def start_simulation(self):
        processes = []

        # Start the server
        server_process = Process(
            target=start_server,
            args=(self.strategy, self.simulation_parameters), daemon=True
        )

        server_process.start()
        processes.append(server_process)

        # Block the script here for a second or two so the server has time to
        # start
        time.sleep(2)

        rng = np.random.default_rng(seed=self.seed)

        for client_identifier in rng.choice(
                self.dataset_factory.get_number_of_clients(),
                size=self.simulation_parameters["num_clients"],
                replace=False):
            client_process = Process(target=start_client,
                                     args=(self.client_provider,
                                           client_identifier),
                                     daemon=True)
            client_process.start()
            processes.append(client_process)

        for p in processes:
            p.join()