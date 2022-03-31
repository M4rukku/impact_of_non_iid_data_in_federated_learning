import flwr as fl
import flwr.client

from sources.flwr_parameters.simulation_parameters import DEFAULT_SERVER_ADDRESS
from sources.simulation_framework.simulators.base_client_provider import BaseClientProvider


def start_client(client_provider: BaseClientProvider, client_identifier):

    client = client_provider(str(client_identifier))
    if isinstance(client, flwr.client.NumPyClient):
        fl.client.start_numpy_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)
    else:
        fl.client.start_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)