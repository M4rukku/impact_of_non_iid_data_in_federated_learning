from collections import defaultdict
from typing import List, Tuple, Union, Optional, Dict

import flwr
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from sources.flwr.flwr_strategies_decorators.base_strategy_decorator import BaseStrategyDecorator

WeightedDictList = List[Tuple[int, Dict[str, Scalar]]]
NumericDict = Dict[str, Union[float, int, complex]]


def check_numeric(x):
    if isinstance(x, (int, float, complex)):
        return True
    else:
        return False


def weighted_average_dict(weighted_dict_list: WeightedDictList) -> NumericDict:
    if len(weighted_dict_list) < 1:
        return {}

    weights_list, metrics_dict_list = list(zip(*weighted_dict_list))
    total_weight = sum(weights_list)
    results_dict = defaultdict(lambda: 0.0)

    # Get all metrics referring to numerics
    relevant_metrics = list(map(lambda p: p[0], filter(lambda p: check_numeric(p[1]), metrics_dict_list[0].items())))

    for weight, metrics_dict in zip(weights_list, metrics_dict_list):
        for metric_name in relevant_metrics:
            results_dict[metric_name] += metrics_dict[metric_name] * weight

    return {key: value / total_weight for key, value in results_dict.items()}


class ImproveAggregatedEvaluationDecorator(BaseStrategyDecorator):

    def __init__(self, strategy: flwr.server.strategy.Strategy, accept_failures=True):
        super().__init__(strategy)
        self.accept_failures = accept_failures

    def aggregate_evaluate(self,
                           rnd: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Union[
        Tuple[Optional[float], Dict[str, Scalar]],
        Optional[float],  # Deprecated
    ]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )

        weighted_metrics_list: WeightedDictList = \
            [(evaluate_res.num_examples, evaluate_res.metrics)
             for _, evaluate_res in results if evaluate_res.metrics is not None]
        metrics_aggregated = weighted_average_dict(
            weighted_metrics_list
        )
        return loss_aggregated, metrics_aggregated
