import tensorflow as tf

from experiments.leaf_experiment_metadata_providers import \
    femnist_medium_experiment_metadata_provider, shakespeare_medium_experiment_metadata_provider, \
    celeba_medium_experiment_metadata_provider


def celeba_initial_iid_experiment_metadata_provider(total_clients: int):
    return [
        celeba_medium_experiment_metadata_provider(total_clients),
        celeba_medium_experiment_metadata_provider(total_clients),
        celeba_medium_experiment_metadata_provider(total_clients)
    ]


def celeba_initial_iid_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.2),
        tf.keras.optimizers.SGD(0.1),
        tf.keras.optimizers.SGD(0.3)
    ]


def femnist_initial_iid_experiment_metadata_provider(total_clients: int):
    return [
        femnist_medium_experiment_metadata_provider(total_clients),
        femnist_medium_experiment_metadata_provider(total_clients),
        femnist_medium_experiment_metadata_provider(total_clients)
    ]


def femnist_initial_iid_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.0003),
        tf.keras.optimizers.SGD(0.0004),
        tf.keras.optimizers.SGD(0.0006)
    ]


def shakespeare_initial_iid_experiment_metadata_provider(total_clients: int):
    return [
        shakespeare_medium_experiment_metadata_provider(total_clients),
        shakespeare_medium_experiment_metadata_provider(total_clients),
        shakespeare_medium_experiment_metadata_provider(total_clients)
    ]


def shakespeare_initial_iid_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.8),
        tf.keras.optimizers.SGD(1.0),
        tf.keras.optimizers.SGD(1.2),
    ]
