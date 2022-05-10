import logging
import math
from typing import Callable, Union, cast

from flwr import common
from flwr.client import Client, NumPyClient
from flwr.client.numpy_client import NumPyClientWrapper
from flwr.common import PropertiesRes
from flwr.server.client_proxy import ClientProxy

from sources.utils.exception_definitions import LossNotDefinedError, LossDivergedError

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

        if "loss" in res.metrics:
            if math.isnan(res.metrics["loss"]):
                logging.warning(f"Error Client {self.cid}'s loss diverged. Raising Exception")
                raise LossDivergedError("Error Client returned fit results without loss")
        else:
            logging.warning(f"Error Client {self.cid} does not have a 'loss' parameter in its "
                            f"metrics dict. If this was intended, please add a dummy loss parameter"
                            f" to the metrics dict returned by fit")
            raise LossNotDefinedError("Error Client returned fit results without loss")

        return cast(
            common.FitRes,
            res,
        )

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        instance: Client = _create_client(self.client_fn, self.cid)
        res = instance.evaluate(ins)

        if "loss" in res.metrics:
            if math.isnan(res.metrics["loss"]):
                logging.warning(f"Error Client {self.cid}'s loss diverged during Evaluation. "
                                f"Raising Exception")
                raise LossDivergedError("Error Client returned evaluation results divergent loss")
        else:
            logging.warning(f"Error Client {self.cid} does not have a 'loss' parameter in its "
                            f"metrics dict. If this was intended, please add a dummy loss parameter"
                            f" to the metrics dict returned by evaluate")
            raise LossNotDefinedError("Error Client returned evaluate results without loss")

        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        return common.Disconnect(reason="")  # Nothing to do here (yet)
