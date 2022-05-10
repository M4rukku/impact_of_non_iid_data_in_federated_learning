from sources.experiments.experiment_metadata_provider_utils import ExperimentMetadataProvider
from sources.flwr.flwr_servers.early_stopping_server import DEFAULT_NUM_ROUNDS_ABOVE_TARGET

GB_NORM_EXPERIMENTS_FIXED_METADATA_C10 = {
    "num_clients": 100,
    "num_rounds": 2500,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": 2,
    "target_accuracy": 0.6,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}
GB_NORM_EXPERIMENTS_BASE_METADATA_C10 = ExperimentMetadataProvider(
    GB_NORM_EXPERIMENTS_FIXED_METADATA_C10)

LENET_GB_NORM_EXPERIMENTS_FIXED_METADATA_C10 = {
    "num_clients": 100,
    "num_rounds": 500,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.6,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}
LENET_GB_NORM_EXPERIMENTS_BASE_METADATA_C10 = ExperimentMetadataProvider(
    LENET_GB_NORM_EXPERIMENTS_FIXED_METADATA_C10)