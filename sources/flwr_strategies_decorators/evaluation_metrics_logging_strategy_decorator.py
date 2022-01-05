from collections import defaultdict
from os import PathLike
import numpy as np
from typing import Optional, Tuple, Dict, List, Union

import flwr.server.strategy
from flwr.common import Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from sources.flwr_parameters.saving_parameters import pickle_parameters_to_file, \
    create_round_based_evaluation_metrics_filename, append_data_to_file, \
    create_evaluation_metrics_filename
from sources.flwr_strategies_decorators.base_strategy_decorator import \
    BaseStrategyDecorator


def reduce_average_metrics(num_examples_list, list_metrics):
    result_dict = defaultdict(lambda: 0.0)

    for num_examples, metric_dict in zip(num_examples_list, list_metrics):
        for key, val in metric_dict.items():
            result_dict[key] += val * num_examples

    total = sum(num_examples_list)
    for key, val in result_dict.items():
        result_dict[key] /= total

    return {key:val for key, val in result_dict.items()}


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

        all_failures = results is None or len(results)==0
        accuracies = None

        if not all_failures:
            first_evaluate_res = results[0][1]
            if "acc" in first_evaluate_res.metrics:
                accuracies = [r.metrics["acc"] * r.num_examples for _, r in results]
            elif "accuracy" in first_evaluate_res.metrics:
                accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]

        if accuracies is not None:
            examples = [r.num_examples for _, r in results]
            accuracy_aggregated = sum(accuracies) / sum(examples)
            append_data_to_file(
                create_evaluation_metrics_filename(self.metrics_logging_folder,
                                                   self.experiment_identifier),
                str(accuracy_aggregated))
        else:
            append_data_to_file(
                create_evaluation_metrics_filename(self.metrics_logging_folder,
                                                   self.experiment_identifier),
                str(np.nan))

        if not all_failures:
            metric_list = list(map(lambda p: p[1].metrics, results))
            num_examples_list = list(map(lambda p: p[1].num_examples, results))
            averaged_metrics = reduce_average_metrics(num_examples_list, metric_list)

            pickle_parameters_to_file(
                create_round_based_evaluation_metrics_filename(rnd,
                                                               self.metrics_logging_folder,
                                                               self.experiment_identifier),
                averaged_metrics)

        return self.strategy.aggregate_evaluate(rnd, results, failures)
