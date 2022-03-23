from typing import Callable, Union, cast

from flwr import common
from flwr.client import Client, NumPyClient
from flwr.client.numpy_client import NumPyClientWrapper
from flwr.common import PropertiesRes
from flwr.server.client_proxy import ClientProxy

ClientFn = Callable[[str], Client]


def _create_client(client_fn: ClientFn, cid: str) -> Client:
    """Create a client instance."""
    client: Union[Client, NumPyClient] = client_fn(cid)
    if isinstance(client, NumPyClient):
        client = NumPyClientWrapper(numpy_client=client)
    return client


class SerialExecutionClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(self, client_fn: ClientFn, cid: str):
        super().__init__(cid)
        self.client_fn = client_fn

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        instance: Client = _create_client(self.client_fn, self.cid)
        res = instance.get_parameters()
        return cast(
            common.ParametersRes,
            res,
        )

    def get_properties(self, ins: common.PropertiesIns) -> PropertiesRes:
        """Returns client's properties."""
        instance: Client = _create_client(self.client_fn, self.cid)
        res = instance.get_properties(ins)
        return cast(
            PropertiesRes,
            res,
        )

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        instance: Client = _create_client(self.client_fn, self.cid)
        res = instance.fit(ins)
        return cast(
            common.FitRes,
            res,
        )

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        instance: Client = _create_client(self.client_fn, self.cid)
        res = instance.evaluate(ins)
        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        return common.Disconnect(reason="")  # Nothing to do here (yet)