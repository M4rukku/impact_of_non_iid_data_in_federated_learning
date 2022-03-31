from os import PathLike
from typing import Optional, Tuple, Dict, Union, List

import flwr
from flwr.common import Parameters, Scalar, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from sources.flwr.flwr_strategies_decorators.saving_strategy_decorators_utils import pickle_parameters_to_file, \
    create_round_based_centralised_evaluation_metrics_filename
from sources.flwr.flwr_strategies_decorators.base_strategy_decorator import BaseStrategyDecorator


class CentralEvaluationLoggingDecorator(BaseStrategyDecorator):

    def __init__(self,
                 strategy: flwr.server.strategy.Strategy,
                 metrics_logging_folder: Union[PathLike, str],
                 experiment_identifier: str,
                 rounds_between_evaluations: int = 10):
        super().__init__(strategy)
        self.rnd = 1
        self.metrics_logging_folder = metrics_logging_folder
        self.experiment_identifier = experiment_identifier
        self.rounds_between_evaluations = rounds_between_evaluations

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        if self.rnd % self.rounds_between_evaluations != 0:
            return None

        response = self.strategy.evaluate(parameters)

        if len(response) == 0:
            return None

        loss, metrics = response
        pickle_parameters_to_file(
            create_round_based_centralised_evaluation_metrics_filename(self.rnd,
                                                                       self.metrics_logging_folder,
                                                                       self.experiment_identifier),
            metrics)
        self.rnd += 1

        return loss, metrics

    def configure_evaluate(self, rnd: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(rnd, parameters, client_manager)