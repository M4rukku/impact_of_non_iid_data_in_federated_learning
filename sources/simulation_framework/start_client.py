import flwr as fl

from sources.flwr_clients.base_client import BaseClient
from sources.flwr_parameters.simulation_parameters import DEFAULT_SERVER_ADDRESS


def start_client(model_template, dataset_factory, client_identifier, metrics,
                 fitting_callbacks, evaluation_callbacks):

    client = BaseClient(model_template,
                        dataset_factory.create_dataset(str(client_identifier)),
                        metrics, fitting_callbacks, evaluation_callbacks)
    fl.client.start_numpy_client(server_address=DEFAULT_SERVER_ADDRESS,
                                 client=client)