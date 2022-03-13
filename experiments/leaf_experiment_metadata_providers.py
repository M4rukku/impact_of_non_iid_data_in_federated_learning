from enum import IntEnum, auto
from typing import Dict

from sources.experiments.experiment_metadata_provider_utils import FixedExperimentMetadata, ExperimentMetadataProvider


class ExperimentScale(IntEnum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()


CELEBA_SCALE_EXPERIMENT_METADATA_MAP: Dict[ExperimentScale, FixedExperimentMetadata] = {
    ExperimentScale.SMALL: {
        "num_clients": None,
        "num_rounds": 30,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 10,
        "val_steps": 2
    },
    ExperimentScale.MEDIUM: {
        "num_clients": None,
        "num_rounds": 100,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 10,
        "val_steps": 2
    },
    ExperimentScale.LARGE: {
        "num_clients": None,
        "num_rounds": 400,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 20,
        "val_steps": 2
    }
}

celeba_small_experiment_metadata_provider = ExperimentMetadataProvider(
    CELEBA_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.SMALL]
)

celeba_medium_experiment_metadata_provider = ExperimentMetadataProvider(
    CELEBA_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.MEDIUM]
)

celeba_large_experiment_metadata_provider = ExperimentMetadataProvider(
    CELEBA_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.LARGE]
)

FEMNIST_SCALE_EXPERIMENT_METADATA_MAP: Dict[ExperimentScale, FixedExperimentMetadata] = {
    ExperimentScale.SMALL: {
        "num_clients": None,
        "num_rounds": 30,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 10,
        "val_steps": 2
    },
    ExperimentScale.MEDIUM: {
        "num_clients": None,
        "num_rounds": 100,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 10,
        "val_steps": 2
    },
    ExperimentScale.LARGE: {
        "num_clients": None,
        "num_rounds": 400,
        "clients_per_round": 3,
        "batch_size": 5,
        "local_epochs": 20,
        "val_steps": 2
    }
}

femnist_small_experiment_metadata_provider = ExperimentMetadataProvider(
    FEMNIST_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.SMALL]
)

femnist_medium_experiment_metadata_provider = ExperimentMetadataProvider(
    FEMNIST_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.MEDIUM]
)

femnist_large_experiment_metadata_provider = ExperimentMetadataProvider(
    FEMNIST_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.LARGE]
)

SHAKESPEARE_SCALE_EXPERIMENT_METADATA_MAP: Dict[ExperimentScale, FixedExperimentMetadata] = {
    ExperimentScale.SMALL: {
        "num_clients": None,
        "num_rounds": 6,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 2,
        "val_steps": 1
    },
    ExperimentScale.MEDIUM: {
        "num_clients": None,
        "num_rounds": 8,
        "clients_per_round": 2,
        "batch_size": 5,
        "local_epochs": 2,
        "val_steps": 1
    },
    ExperimentScale.LARGE: {
        "num_clients": None,
        "num_rounds": 20,
        "clients_per_round": 3,
        "batch_size": 5,
        "local_epochs": 1,
        "val_steps": 1
    }
}

shakespeare_small_experiment_metadata_provider = ExperimentMetadataProvider(
    SHAKESPEARE_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.SMALL]
)

shakespeare_medium_experiment_metadata_provider = ExperimentMetadataProvider(
    SHAKESPEARE_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.MEDIUM]
)

shakespeare_large_experiment_metadata_provider = ExperimentMetadataProvider(
    SHAKESPEARE_SCALE_EXPERIMENT_METADATA_MAP[ExperimentScale.LARGE]
)