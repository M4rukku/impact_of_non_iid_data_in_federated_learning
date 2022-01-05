import dataclasses
from dataclasses import dataclass
from typing import Union, Dict
import tensorflow as tf

from sources.experiments.experiment_metadata import ExperimentMetadata


@dataclass
class ExtendedExperimentMetadata:
    strategy_name: str
    optimizer_config: Dict
    clients_per_round: Union[float, int]
    num_clients: int
    num_rounds: int
    batch_size: int
    local_epochs: int
    val_steps: int
    local_learning_rate: Union[float, None] = None


def create_extended_experiment_metadata(
        experiment_metadata: ExperimentMetadata,
        strategy_name: str,
        optimizer: tf.keras.optimizers.Optimizer):
    lr = optimizer.__dict__["learning_rate"] if "learning_rate" in optimizer.__dict__ else None

    return ExtendedExperimentMetadata(
        strategy_name=strategy_name,
        optimizer_config=optimizer.get_config(),
        local_learning_rate=lr,
        **dataclasses.asdict(experiment_metadata)
    )
