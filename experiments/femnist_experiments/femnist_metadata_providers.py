from sources.experiments.experiment_metadata_provider_utils import ExperimentMetadataProvider
from sources.global_data_properties import FEMNIST_CLIENTS_TO_CONSIDER
from sources.flwr.flwr_servers.early_stopping_server import DEFAULT_NUM_ROUNDS_ABOVE_TARGET

FEMNIST_FIXED_METADATA_SYS_EXP = {
    "num_clients": FEMNIST_CLIENTS_TO_CONSIDER,
    "num_rounds": 500,
    "clients_per_round": 10,
    "batch_size": 10,
    "local_epochs": 1,
    "val_steps": 2,
    "target_accuracy": None,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}
FEMNIST_BASE_METADATA_SYS_EXP_PROVIDER = ExperimentMetadataProvider(FEMNIST_FIXED_METADATA_SYS_EXP)

VARYING_REPORTING_FRACTION_EXP_PARAMETER_MAP = {"clients_per_round": [5, 10, 20, 40]}
VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP = {"local_epochs": [1, 5, 10, 20]}

##############################################################################################
# For Hyperparameter Search

FEMNIST_FIXED_METADATA_HYPERPARAMETER_SEARCH = {
    "num_clients": FEMNIST_CLIENTS_TO_CONSIDER,
    "num_rounds": 200,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": 1,
    "target_accuracy": 0.6,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

FEMNIST_BASE_METADATA_HYPERPARAMETER_SEARCH_PROVIDER = \
    ExperimentMetadataProvider(FEMNIST_FIXED_METADATA_HYPERPARAMETER_SEARCH)



# For Varying the Opimiser Experiments
FEMNIST_FIXED_METADATA_OPT_EXP = {
    "num_clients": FEMNIST_CLIENTS_TO_CONSIDER,
    "num_rounds": 750,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": None,
    "target_accuracy": 0.7,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

FEMNIST_BASE_METADATA_OPT_EXP_PROVIDER = ExperimentMetadataProvider(FEMNIST_FIXED_METADATA_OPT_EXP)

#####################################################################################

# For Varying the Remaining Experiments - Small IID + Proximal Term
FEMNIST_FIXED_METADATA_REM_EXP = {
    "num_clients": FEMNIST_CLIENTS_TO_CONSIDER,
    "num_rounds": 500,
    "clients_per_round": 10,
    "batch_size": 10,
    "local_epochs": 1,
    "val_steps": 2,
    "target_accuracy": 0.6,
    "num_rounds_above_target": DEFAULT_NUM_ROUNDS_ABOVE_TARGET
}

FEMNIST_BASE_METADATA_REM_EXP_PROVIDER = ExperimentMetadataProvider(FEMNIST_FIXED_METADATA_REM_EXP)
