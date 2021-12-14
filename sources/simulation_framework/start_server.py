import flwr as fl

from sources.flwr_parameters.simulation_parameters import SimulationParameters, \
    DEFAULT_SERVER_ADDRESS


def start_server(strategy: fl.server.strategy.Strategy,
                 simulation_parameters: SimulationParameters):
    fl.server.start_server(server_address=DEFAULT_SERVER_ADDRESS,
                           strategy=strategy,
                           config={"num_rounds": simulation_parameters[
                               "num_rounds"]})