import functools
import math

import numpy
import tensorflow as tf

from experiments.leaf_experiment_metadata_providers import \
    celeba_medium_experiment_metadata_provider, \
    femnist_medium_experiment_metadata_provider, \
    shakespeare_medium_experiment_metadata_provider

from sources.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_yogi_strategy_provider, full_eval_fed_adagrad_strategy_provider, \
    full_eval_fed_adam_strategy_provider, full_eval_fed_avg_strategy_provider


def celeba_vom_experiment_metadata_provider(total_clients: int):
    return [
        celeba_medium_experiment_metadata_provider(total_clients),
        celeba_medium_experiment_metadata_provider(total_clients),
        celeba_medium_experiment_metadata_provider(total_clients),
        celeba_medium_experiment_metadata_provider(total_clients)
    ]


def celeba_vom_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.1),
        tf.keras.optimizers.SGD(0.1),
        tf.keras.optimizers.SGD(0.1),
        tf.keras.optimizers.SGD(0.1)
    ]


def femnist_vom_experiment_metadata_provider(total_clients: int):
    return [
        femnist_medium_experiment_metadata_provider(total_clients),
        femnist_medium_experiment_metadata_provider(total_clients),
        femnist_medium_experiment_metadata_provider(total_clients),
        femnist_medium_experiment_metadata_provider(total_clients)
    ]


def femnist_vom_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.0004),
        tf.keras.optimizers.SGD(0.0004),
        tf.keras.optimizers.SGD(0.0004),
        tf.keras.optimizers.SGD(0.0004)
    ]


def shakespeare_vom_experiment_metadata_provider(total_clients: int):
    return [
        shakespeare_medium_experiment_metadata_provider(total_clients),
        shakespeare_medium_experiment_metadata_provider(total_clients),
        shakespeare_medium_experiment_metadata_provider(total_clients),
        shakespeare_medium_experiment_metadata_provider(total_clients)
    ]


def shakespeare_vom_experiment_optimizer_provider():
    return [
        tf.keras.optimizers.SGD(0.8),
        tf.keras.optimizers.SGD(0.8),
        tf.keras.optimizers.SGD(0.8),
        tf.keras.optimizers.SGD(0.8),
    ]


def vom_experiments_strategy_providers(eval_fn,
                                       initial_parameters: numpy.array,
                                       eta: float = math.pow(10, -3/2),
                                       eta_l: float = math.pow(10, -3/2),
                                       tau: float = 1e-2
                                       ):
    return [
        functools.partial(full_eval_fed_avg_strategy_provider,
                          eval_fn=eval_fn
                          ),
        functools.partial(full_eval_fed_adagrad_strategy_provider,
                          eval_fn=eval_fn,
                          initial_parameters=initial_parameters,
                          eta=eta, eta_l=eta_l, tau=tau,
                          ),
        functools.partial(full_eval_fed_adam_strategy_provider,
                          eval_fn=eval_fn,
                          initial_parameters=initial_parameters,
                          eta=eta, eta_l=eta_l, tau=tau,
                          beta_1=0.9, beta_2=0.99
                          ),
        functools.partial(full_eval_fed_yogi_strategy_provider,
                          eval_fn=eval_fn,
                          initial_parameters=initial_parameters,
                          eta=eta, eta_l=eta_l, tau=tau,
                          beta_1=0.9, beta_2=0.99
                          )
    ]
