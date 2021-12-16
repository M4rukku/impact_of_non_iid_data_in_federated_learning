from os import PathLike
from typing import Optional, Tuple, Dict, List, Union

import flwr.server.strategy
from flwr.common import Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from sources.flwr_parameters.saving_parameters import pickle_parameters_to_file, \
    create_round_based_evaluation_metrics_filename, append_data_to_file, \
    create_evaluation_metrics_filename
from sources.flwr_strategies.base_strategy_decorator import \
    BaseStrategyDecorator


class EvaluationMetricsLoggingStrategyDecorator(BaseStrategyDecorator):

    def __init__(self,
                 strategy: flwr.server.strategy.Strategy,
                 metrics_logging_folder: Union[PathLike, str],
                 experiment_identifier: str):
        super().__init__(strategy)
        self.metrics_logging_folder = metrics_logging_folder
        self.experiment_identifier = experiment_identifier

    def aggregate_evaluate(self,
                           rnd: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]
                           ) -> Union[
        Tuple[Optional[float], Dict[str, Scalar]], Optional[float]]:

        accuracies = None
        if "acc" in results[0][1].metrics:
            accuracies = [r.metrics["acc"] * r.num_examples for _, r in results]
        elif "accuracy" in results[0][1].metrics:
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in
                          results]

        if accuracies is not None:
            examples = [r.num_examples for _, r in results]
            accuracy_aggregated = sum(accuracies) / sum(examples)
            append_data_to_file(
                create_evaluation_metrics_filename(self.metrics_logging_folder,
                                                   self.experiment_identifier),
                str(accuracy_aggregated))

        metric_parameters = list(map(lambda p: p[1].metrics, results))

        pickle_parameters_to_file(
            create_round_based_evaluation_metrics_filename(rnd,
                                                           self.metrics_logging_folder,
                                                           self.experiment_identifier),
            metric_parameters)

        return self.strategy.aggregate_evaluate(rnd, results, failures)
