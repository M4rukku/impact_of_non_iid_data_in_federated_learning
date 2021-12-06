from os import PathLike
from typing import Optional, Tuple, Dict, List, Union

import flwr.server.strategy
from flwr.common import Parameters, Scalar, FitRes, Weights
from flwr.server.client_proxy import ClientProxy

from sources.flwr_parameters.saving_parameters import create_round_based_model_saving_filename, npz_parameters_to_file
from sources.flwr_strategies.base_strategy_decorator import BaseStrategyDecorator


class ModelLoggingStrategyDecorator(BaseStrategyDecorator):

    def __init__(self, strategy: flwr.server.strategy.Strategy,
                 model_saving_folder: PathLike[str],
                 experiment_identifier: str):
        super().__init__(strategy)
        self.model_saving_folder = model_saving_folder
        self.experiment_identifier = experiment_identifier

    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> \
            Union[
                Tuple[Optional[Parameters], Dict[str, Scalar]],
                Optional[Weights],  # Deprecated
            ]:
        aggregated_weights = self.strategy.aggregate_fit(rnd, results, failures)
        npz_parameters_to_file(create_round_based_model_saving_filename(rnd,
                                                                        self.model_saving_folder,
                                                                        self.experiment_identifier),
                               aggregated_weights)
        return aggregated_weights
