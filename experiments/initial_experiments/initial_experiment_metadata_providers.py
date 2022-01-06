import tensorflow as tf

from experiments.leaf_experiment_metadata_providers import \
    celeba_small_experiment_metadata_provider, celeba_medium_experiment_metadata_provider, \
    celeba_large_experiment_metadata_provider, femnist_small_experiment_metadata_provider, \
    femnist_medium_experiment_metadata_provider, femnist_large_experiment_metadata_provider, \
    shakespeare_large_experiment_metadata_provider, \
    shakespeare_medium_experiment_metadata_provider, \
    shakespeare_small_experiment_metadata_provider


def celeba_initial_experiment_metadata_provider(total_clients: int):
    return [
        celeba_small_experiment_metadata_provider(total_clients),
        celeba_medium_experiment_metadata_provider(total_clients),
        celeba_large_experiment_metadata_provider(total_clients)
    ]


def celeba_initial_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.1),
        tf.keras.optimizers.SGD(0.1),
        tf.keras.optimizers.SGD(0.1)
    ]


def femnist_initial_experiment_metadata_provider(total_clients: int):
    return [
        femnist_small_experiment_metadata_provider(total_clients),
        femnist_medium_experiment_metadata_provider(total_clients),
        femnist_large_experiment_metadata_provider(total_clients)
    ]


def femnist_initial_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.0004),
        tf.keras.optimizers.SGD(0.0004),
        tf.keras.optimizers.SGD(0.0004)
    ]


def shakespeare_initial_experiment_metadata_provider(total_clients: int):
    return [
        shakespeare_small_experiment_metadata_provider(total_clients),
        shakespeare_medium_experiment_metadata_provider(total_clients),
        shakespeare_large_experiment_metadata_provider(total_clients)
    ]


def shakespeare_initial_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.8),
        tf.keras.optimizers.SGD(0.8),
        tf.keras.optimizers.SGD(0.8),
    ]
