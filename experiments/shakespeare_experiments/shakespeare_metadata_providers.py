from sources.experiments.experiment_metadata_provider_utils import ExperimentMetadataProvider, \
    FixedExperimentMetadataEarlyStopping
from sources.global_data_properties import SHAKESPEARE_CLIENTS_TO_CONSIDER
from sources.flwr.flwr_servers.early_stopping_server import DEFAULT_NUM_ROUNDS_ABOVE_TARGET

SHAKESPEARE_FIXED_METADATA_SYS_EXP: FixedExperimentMetadataEarlyStopping = {
    "num_clients": SHAKESPEARE_CLIENTS_TO_CONSIDER,
    "num_rounds": 40,
    "clients_per_round": 10,
    "batch_size": 4,
    "local_epochs": 1,
    "val_steps": 100,
    "target_accuracy": 0.5,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

SHAKESPEARE_BASE_METADATA_SYS_EXP_PROVIDER = ExperimentMetadataProvider(
    SHAKESPEARE_FIXED_METADATA_SYS_EXP)

VARYING_REPORTING_FRACTION_EXP_PARAMETER_MAP = {"clients_per_round": [1, 5, 10]}
VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP = {"local_epochs": [1, 5, 10, 20]}

##############################################################################################
# For Hyperparameter Search

SHAKESPEARE_FIXED_METADATA_HYPERPARAMETER_SEARCH: FixedExperimentMetadataEarlyStopping = {
    "num_clients": SHAKESPEARE_CLIENTS_TO_CONSIDER,
    "num_rounds": 25,
    "clients_per_round": 10,
    "batch_size": 5,
    "local_epochs": 1,
    "val_steps": 10,
    "target_accuracy": 0.5,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

SHAKESPEARE_BASE_METADATA_HYPERPARAMETER_SEARCH_PROVIDER = \
    ExperimentMetadataProvider(SHAKESPEARE_FIXED_METADATA_HYPERPARAMETER_SEARCH)

# For Varying the Opimiser Experiments
SHAKESPEARE_FIXED_METADATA_OPT_EXP = {
    "num_clients": SHAKESPEARE_CLIENTS_TO_CONSIDER,
    "num_rounds": 40,
    "clients_per_round": 10,
    "batch_size": 4,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.5,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

SHAKESPEARE_BASE_METADATA_OPT_EXP_PROVIDER = ExperimentMetadataProvider(
    SHAKESPEARE_FIXED_METADATA_OPT_EXP)

#####################################################################################

# For Varying the Remaining Experiments - Small IID + Proximal Term
SHAKESPEARE_FIXED_METADATA_REM_EXP = {
    "num_clients": SHAKESPEARE_CLIENTS_TO_CONSIDER,
    "num_rounds": 40,
    "clients_per_round": 10,
    "batch_size": 4,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.5,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

SHAKESPEARE_BASE_METADATA_REM_EXP_PROVIDER = ExperimentMetadataProvider(
    SHAKESPEARE_FIXED_METADATA_REM_EXP)
