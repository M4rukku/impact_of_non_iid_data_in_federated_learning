import copy
import dataclasses
import itertools
import math
from typing import Dict, Callable, List, Optional

from flwr.server.strategy import Strategy
from tensorflow_addons.utils.types import Optimizer

from sources.experiments.experiment_metadata import ExperimentMetadata


@dataclasses.dataclass
class ParameterGridResponse:
    strategy_provider_list: List[Callable[[ExperimentMetadata], Strategy]]
    experiment_metadata_list: List[ExperimentMetadata]
    optimizer_list: List[Optimizer]


def default_suffix_provider(parameter_value_map: Dict[str, float],
                            log10_representation=True):
    if log10_representation:
        parameter_value_map = {key: math.log10(val) for key, val in parameter_value_map.items()}
    return "_".join([f"{str(key)}{val:.2f}" for key, val in parameter_value_map.items()])


class ParameterGridMetadataGenerator:

    def __init__(self,
                 parameter_value_map: Dict[str, List[float]],
                 strategy_provider_function: Callable[[Dict[str, float]],
                                                      Callable[[ExperimentMetadata], Strategy]],
                 optimizer_provider_function: Callable[[Dict[str, float]], Optimizer],
                 base_experiment_metadata: ExperimentMetadata,
                 custom_suffix_provider: Optional[Callable[[Dict[str, List[float]]],
                                                           str]] = None
                 ):
        self.parameter_value_map = parameter_value_map
        self.strategy_provider_function = strategy_provider_function
        self.optimizer_provider_function = optimizer_provider_function
        self.base_experiment_metadata = base_experiment_metadata
        self.custom_suffix_provider = custom_suffix_provider

    def generate_grid_responses(self) -> List[ParameterGridResponse]:
        order = self.parameter_value_map.keys()
        pools = [self.parameter_value_map[key] for key in order]
        products = itertools.product(*pools)
        response = ParameterGridResponse([], [], [])

        for product in products:
            current_parameter_dict = {key: val for key, val in zip(order, product)}
            strategy = self.strategy_provider_function(current_parameter_dict)
            experiment_metadata = copy.deepcopy(self.base_experiment_metadata)

            if self.custom_suffix_provider is not None:
                experiment_metadata.custom_suffix = self.custom_suffix_provider(
                    current_parameter_dict
                )

            optimizer = self.optimizer_provider_function(current_parameter_dict)

            response.strategy_provider_list.append(strategy)
            response.experiment_metadata_list.append(experiment_metadata)
            response.optimizer_list.append(optimizer)

        return response
