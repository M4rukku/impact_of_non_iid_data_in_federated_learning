from os import PathLike
from typing import Optional, Tuple, Dict, Union

import flwr
from flwr.common import Parameters, Scalar

from sources.flwr_parameters.saving_parameters import pickle_parameters_to_file, \
    create_round_based_centralised_evaluation_metrics_filename
from sources.flwr_strategies.base_strategy_decorator import BaseStrategyDecorator


class CentralEvaluationLoggingDecorator(BaseStrategyDecorator):

    def __init__(self,
                 strategy: flwr.server.strategy.Strategy,
                 metrics_logging_folder: Union[PathLike, str],
                 experiment_identifier: str):
        super().__init__(strategy)
        self.rnd = 1
        self.metrics_logging_folder = metrics_logging_folder
        self.experiment_identifier = experiment_identifier

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        loss, metrics = self.strategy.evaluate(parameters)

        pickle_parameters_to_file(
            create_round_based_centralised_evaluation_metrics_filename(self.rnd,
                                                                       self.metrics_logging_folder,
                                                                       self.experiment_identifier),
            metrics)
        self.rnd += 1

        return loss, metrics
