from sources.experiments.experiment_metadata_provider_utils import ExperimentMetadataProvider

CIFAR10_FIXED_METADATA_SYS_EXP = {
    "num_clients": 100,
    "num_rounds": 2500,
    "clients_per_round": 10,
    "batch_size": 64,
    "local_epochs": 1,
    "val_steps": 1,
    "target_accuracy": 0.7,
    "num_rounds_above_target": 3
}
CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER = ExperimentMetadataProvider(CIFAR10_FIXED_METADATA_SYS_EXP)

VARYING_REPORTING_FRACTION_EXP_PARAMETER_MAP = {"clients_per_round": [5, 10, 20, 40]}
VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP = {"local_epochs": [1, 5, 10, 20]}

##############################################################################################

# For Varying the Opimiser Experiments
CIFAR10_FIXED_METADATA_OPT_EXP = {
    "num_clients": 100,
    "num_rounds": 5000,
    "clients_per_round": 10,
    "batch_size": 20,
    "local_epochs": 1,
    "val_steps": 1,
    "target_accuracy": 0.7,
    "num_rounds_above_target": 3
}

CIFAR10_BASE_METADATA_OPT_EXP_PROVIDER = ExperimentMetadataProvider(CIFAR10_FIXED_METADATA_OPT_EXP)

#####################################################################################

# For Varying the Remaining Experiments - Small IID + Proximal Term
CIFAR10_FIXED_METADATA_REM_EXP = {
    "num_clients": 100,
    "num_rounds": 2500,
    "clients_per_round": 10,
    "batch_size": 10,
    "local_epochs": 1,
    "val_steps": 1,
    "target_accuracy": 0.7,
    "num_rounds_above_target": 3
}

CIFAR10_BASE_METADATA_REM_EXP_PROVIDER = ExperimentMetadataProvider(CIFAR10_FIXED_METADATA_REM_EXP)