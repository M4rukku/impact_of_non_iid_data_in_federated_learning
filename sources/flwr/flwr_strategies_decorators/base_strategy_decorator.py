from typing import Optional, Tuple, Dict, List, Union

import flwr.server.strategy
from flwr.common import Parameters, Scalar, EvaluateRes, EvaluateIns, FitRes, Weights, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


def get_name_of_strategy(strategy: flwr.server.strategy.Strategy):
    if hasattr(strategy, "strategy_name"):
        return strategy.strategy_name
    else:
        return type(strategy).__name__


class BaseStrategyDecorator(flwr.server.strategy.Strategy):

    def __init__(self, strategy: flwr.server.strategy.Strategy):
        self.strategy = strategy
        self.strategy_name = self.annotate_strategy_name(get_name_of_strategy(strategy))
        self.rnd = 1

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(self, rnd: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, FitIns]]:
        self.rnd = rnd
        return self.strategy.configure_fit(rnd, parameters, client_manager)

    def aggregate_fit(self, rnd: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> \
            Union[
                Tuple[Optional[Parameters], Dict[str, Scalar]],
                Optional[Weights],  # Deprecated
            ]:
        self.rnd = rnd
        return self.strategy.aggregate_fit(rnd, results, failures)

    def configure_evaluate(self, rnd: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, EvaluateIns]]:
        self.rnd = rnd
        return self.strategy.configure_evaluate(rnd, parameters, client_manager)

    def aggregate_evaluate(self,
                           rnd: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Union[
        Tuple[Optional[float], Dict[str, Scalar]],
        Optional[float],  # Deprecated
    ]:
        self.rnd = rnd
        return self.strategy.aggregate_evaluate(rnd, results, failures)

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(parameters)

    def annotate_strategy_name(self, strategy_name: str) -> str:
        return strategy_name
