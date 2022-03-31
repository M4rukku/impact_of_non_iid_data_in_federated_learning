import flwr as fl
from flwr.server import SimpleClientManager

from sources.flwr_parameters.simulation_parameters import SimulationParameters, \
    DEFAULT_SERVER_ADDRESS, EarlyStoppingSimulationParameters
from sources.flwr.flwr_servers.early_stopping_server import EarlyStoppingServer


def start_server(strategy: fl.server.strategy.Strategy,
                 simulation_parameters: SimulationParameters):
    if "target_accuracy" in simulation_parameters and \
            simulation_parameters["target_accuracy"] is not None:
        simulation_parameters: EarlyStoppingSimulationParameters = simulation_parameters
        server = EarlyStoppingServer(SimpleClientManager(),
                                     strategy,
                                     simulation_parameters["target_accuracy"],
                                     simulation_parameters["num_rounds_above_target"]
                                     )
    else:
        server = None

    fl.server.start_server(server_address=DEFAULT_SERVER_ADDRESS,
                           strategy=strategy,
                           server=server,
                           config={"num_rounds": simulation_parameters["num_rounds"]})
