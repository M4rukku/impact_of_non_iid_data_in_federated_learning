import logging
from typing import Optional, Tuple, Dict, List, Union

import flwr.server.strategy
from flwr.common import Parameters, Scalar, EvaluateRes, EvaluateIns, FitRes, Weights, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from sources.flwr.flwr_strategies_decorators.base_strategy_decorator import BaseStrategyDecorator


class StrategyLoggingDecorator(BaseStrategyDecorator):

    def __init__(self, strategy: flwr.server.strategy.Strategy):
        super().__init__(strategy)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        logging.warning("Starting to initialize parameters")
        response = self.strategy.initialize_parameters(client_manager)
        logging.warning("Finishing initializing parameters")
        return response

    def configure_fit(self, rnd: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, FitIns]]:
        logging.warning("Starting to configure fit")
        response = self.strategy.configure_fit(rnd, parameters, client_manager)
        logging.warning("Finishing configuring fit")
        return response

    def aggregate_fit(self, rnd: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> \
            Union[
                Tuple[Optional[Parameters], Dict[str, Scalar]],
                Optional[Weights],  # Deprecated
            ]:
        logging.warning("Starting to aggregate fit")
        response = self.strategy.aggregate_fit(rnd, results, failures)
        logging.warning("Finishing aggregating fit")
        return response

    def configure_evaluate(self, rnd: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, EvaluateIns]]:
        logging.warning("Starting to configure evaluate")
        response = self.strategy.configure_evaluate(rnd, parameters, client_manager)
        logging.warning("Finishing configuring evaluate")
        return response

    def aggregate_evaluate(self,
                           rnd: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Union[
        Tuple[Optional[float], Dict[str, Scalar]],
        Optional[float],  # Deprecated
    ]:
        logging.warning("Starting to aggregate evaluate")
        response = self.strategy.aggregate_evaluate(rnd, results, failures)
        logging.warning("Finishing aggregating evaluation")
        return response

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        logging.warning("Starting to evaluate")
        response = self.strategy.evaluate(parameters)
        logging.warning("Finishing evaluations")
        return response
