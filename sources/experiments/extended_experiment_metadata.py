import dataclasses
from dataclasses import dataclass
from typing import Union, Dict, Optional, Any
import tensorflow as tf

from sources.experiments.experiment_metadata import ExperimentMetadata
from sources.flwr.flwr_servers.early_stopping_server import DEFAULT_NUM_ROUNDS_ABOVE_TARGET


@dataclass
class ExtendedExperimentMetadata:
    strategy_name: str
    optimizer_config: Union[str, Dict[str, Any]]
    clients_per_round: Union[float, int]
    num_clients: int
    num_rounds: int
    batch_size: int
    local_epochs: int
    val_steps: int
    local_learning_rate: Union[float, None] = None
    target_accuracy: Optional[float] = None
    num_rounds_above_target: int = DEFAULT_NUM_ROUNDS_ABOVE_TARGET
    custom_suffix: Optional[str] = None


def create_extended_experiment_metadata(
        experiment_metadata: ExperimentMetadata,
        strategy_name: str,
        optimizer_config: Union[str, Dict[str, Any]],
        optimizer=None):

    if optimizer is not None:
        lr = optimizer.__dict__["learning_rate"] if "learning_rate" in optimizer.__dict__ else None
        if lr is None and isinstance(optimizer_config, dict):
            lr = optimizer_config["learning_rate"] if "learning_rate" in optimizer_config else None
    else:
        lr = None

    return ExtendedExperimentMetadata(
        strategy_name=strategy_name,
        optimizer_config=optimizer_config,
        local_learning_rate=lr,
        **dataclasses.asdict(experiment_metadata)
    )
