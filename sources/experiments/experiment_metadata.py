from dataclasses import dataclass
from typing import Union, Optional
from sources.utils.simulation_parameters import SimulationParameters, EarlyStoppingSimulationParameters
from sources.flwr.flwr_servers.early_stopping_server import DEFAULT_NUM_ROUNDS_ABOVE_TARGET


@dataclass
class ExperimentMetadata:
    num_clients: int
    num_rounds: int
    clients_per_round: Union[float, int]
    batch_size: int
    local_epochs: int
    val_steps: int
    target_accuracy: Optional[float] = None
    num_rounds_above_target: int = DEFAULT_NUM_ROUNDS_ABOVE_TARGET
    custom_suffix: Optional[str] = None


def get_simulation_parameters_from_experiment_metadata(experiment_metadata: ExperimentMetadata) \
        -> Union[SimulationParameters, EarlyStoppingSimulationParameters]:
    if experiment_metadata.target_accuracy is not None:
        return_dict: EarlyStoppingSimulationParameters = {
            "num_clients": experiment_metadata.num_clients,
            "num_rounds": experiment_metadata.num_rounds,
            "target_accuracy": experiment_metadata.target_accuracy,
            "num_rounds_above_target": experiment_metadata.num_rounds_above_target
        }
        return return_dict
    else:
        return {
            "num_clients": experiment_metadata.num_clients,
            "num_rounds": experiment_metadata.num_rounds,
        }
