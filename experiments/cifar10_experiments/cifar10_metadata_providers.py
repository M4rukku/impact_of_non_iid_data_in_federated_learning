from sources.experiments.experiment_metadata_provider_utils import ExperimentMetadataProvider
from sources.flwr.flwr_servers.early_stopping_server import DEFAULT_NUM_ROUNDS_ABOVE_TARGET

CIFAR10_FIXED_METADATA_SYS_EXP = {
    "num_clients": 100,
    "num_rounds": 500,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.6,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}
CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER = ExperimentMetadataProvider(CIFAR10_FIXED_METADATA_SYS_EXP)

VARYING_REPORTING_FRACTION_EXP_PARAMETER_MAP = {"clients_per_round": [1, 5, 10, 20]}
VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP = {"local_epochs": [1, 5, 10, 20]}

##############################################################################################

# For Varying the Opimiser Experiments
CIFAR10_FIXED_METADATA_OPT_EXP = {
    "num_clients": 100,
    "num_rounds": 500,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.7,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

CIFAR10_BASE_METADATA_OPT_EXP_PROVIDER = ExperimentMetadataProvider(CIFAR10_FIXED_METADATA_OPT_EXP)

#####################################################################################

# For Varying the Remaining Experiments - Small IID + Proximal Term
CIFAR10_FIXED_METADATA_REM_EXP = {
    "num_clients": 100,
    "num_rounds": 500,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.6,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

CIFAR10_BASE_METADATA_REM_EXP_PROVIDER = ExperimentMetadataProvider(CIFAR10_FIXED_METADATA_REM_EXP)
