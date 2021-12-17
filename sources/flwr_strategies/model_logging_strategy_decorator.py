from os import PathLike
from typing import Optional, Tuple, Dict, List, Union

import flwr.server.strategy
from flwr.common import Parameters, Scalar, FitRes, Weights
from flwr.server.client_proxy import ClientProxy

from sources.flwr_parameters.saving_parameters import \
    create_round_based_model_saving_filename, npz_parameters_to_file
from sources.flwr_strategies.base_strategy_decorator import \
    BaseStrategyDecorator


def log_every_tenth_round(rnd: int):
    return rnd % 10 == 0


class ModelLoggingStrategyDecorator(BaseStrategyDecorator):

    def __init__(self, strategy: flwr.server.strategy.Strategy,
                 model_saving_folder: Union[PathLike, str],
                 experiment_identifier: str,
                 round_predicate=log_every_tenth_round
                 ):
        super().__init__(strategy)
        self.model_saving_folder = model_saving_folder
        self.experiment_identifier = experiment_identifier
        self.round_predicate = round_predicate

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> \
            Union[
                Tuple[Optional[Parameters], Dict[str, Scalar]],
                Optional[Weights],  # Deprecated
            ]:
        aggregated_weights, metrics = \
            self.strategy.aggregate_fit(rnd, results, failures)

        if self.round_predicate(rnd):
            filename = \
                create_round_based_model_saving_filename(rnd,
                                                         self.model_saving_folder,
                                                         self.experiment_identifier)

            npz_parameters_to_file(filename, aggregated_weights)

        return aggregated_weights, metrics
