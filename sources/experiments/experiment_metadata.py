from dataclasses import dataclass
from typing import Union, Optional
from sources.flwr_parameters.simulation_parameters import SimulationParameters


@dataclass
class ExperimentMetadata:
    num_clients: int
    num_rounds: int
    clients_per_round: Union[float, int]
    batch_size: int
    local_epochs: int
    val_steps: int
    custom_suffix: Optional[str] = None


def get_simulation_parameters_from_experiment_metadata(experiment_metadata: ExperimentMetadata) \
        -> SimulationParameters:
    return {
        "num_clients": experiment_metadata.num_clients,
        "num_rounds": experiment_metadata.num_rounds,
    }
