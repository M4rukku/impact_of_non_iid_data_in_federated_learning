from typing import Callable, Optional
import numpy
from flwr.common import weights_to_parameters, Parameters
from flwr.server.strategy import FedAvg, Strategy, FedYogi, FedAdagrad, FedAdam
from sources.experiments.experiment_metadata import ExperimentMetadata
from sources.flwr.flwr_strategies_decorators.enable_full_evaluation_decorator import \
    EnableFullEvaluationDecorator
from sources.flwr.flwr_strategies_decorators.improve_aggregated_evaluation_decorator import \
    ImproveAggregatedEvaluationDecorator


def get_fraction_fit_from_metadata(experiment_metadata: ExperimentMetadata):
    if experiment_metadata.clients_per_round >= 1:
        fraction_fit = float(experiment_metadata.clients_per_round) / \
                       float(experiment_metadata.num_clients)
    else:
        fraction_fit = experiment_metadata.clients_per_round

    return fraction_fit


def full_eval_fed_avg_strategy_provider(
        eval_fn: Callable,
        experiment_metadata: ExperimentMetadata,
        initial_parameters: Optional[Parameters] = None,
        accept_failures=True,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        **kwargs
) -> Strategy:
    fraction_fit = get_fraction_fit_from_metadata(experiment_metadata)

    strategy = ImproveAggregatedEvaluationDecorator(
        EnableFullEvaluationDecorator(
            FedAvg(
                eval_fn=eval_fn,
                fraction_fit=fraction_fit,
                fraction_eval=fraction_fit,
                initial_parameters=initial_parameters,
                accept_failures=accept_failures,
                min_fit_clients=min_fit_clients,
                min_eval_clients=min_eval_clients,
                **kwargs
            )
        ),
        accept_failures=accept_failures
    )

    return strategy


def full_eval_fed_adam_strategy_provider(
        eval_fn: Callable,
        experiment_metadata: ExperimentMetadata,
        initial_parameters: numpy.array,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        accept_failures=True,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        **kwargs
) -> Strategy:
    fraction_fit = get_fraction_fit_from_metadata(experiment_metadata)
    initial_parameters = weights_to_parameters(initial_parameters)

    strategy = ImproveAggregatedEvaluationDecorator(
        EnableFullEvaluationDecorator(
            FedAdam(
                eval_fn=eval_fn,
                fraction_fit=fraction_fit,
                fraction_eval=fraction_fit,
                eta=eta,
                eta_l=eta_l,
                tau=tau,
                beta_1=beta_1,
                beta_2=beta_2,
                initial_parameters=initial_parameters,
                accept_failures=accept_failures,
                min_fit_clients=min_fit_clients,
                min_eval_clients=min_eval_clients,
                **kwargs
            )
        ),
        accept_failures=accept_failures
    )

    return strategy


def full_eval_fed_adagrad_strategy_provider(
        eval_fn: Callable,
        experiment_metadata: ExperimentMetadata,
        initial_parameters: numpy.array,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-3,
        accept_failures=True,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        **kwargs

) -> Strategy:
    fraction_fit = get_fraction_fit_from_metadata(experiment_metadata)
    initial_parameters = weights_to_parameters(initial_parameters)

    strategy = ImproveAggregatedEvaluationDecorator(
        EnableFullEvaluationDecorator(
            FedAdagrad(
                eval_fn=eval_fn,
                fraction_fit=fraction_fit,
                fraction_eval=fraction_fit,
                eta=eta,
                eta_l=eta_l,
                tau=tau,
                initial_parameters=initial_parameters,
                accept_failures=accept_failures,
                min_fit_clients=min_fit_clients,
                min_eval_clients=min_eval_clients,
                **kwargs
            )
        ),
        accept_failures=accept_failures
    )

    return strategy


def full_eval_fed_yogi_strategy_provider(
        eval_fn: Callable,
        experiment_metadata: ExperimentMetadata,
        initial_parameters: numpy.array,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        accept_failures=True,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        **kwargs
) -> Strategy:
    fraction_fit = get_fraction_fit_from_metadata(experiment_metadata)
    initial_parameters = weights_to_parameters(initial_parameters)

    strategy = ImproveAggregatedEvaluationDecorator(
        EnableFullEvaluationDecorator(
            FedYogi(
                eval_fn=eval_fn,
                fraction_fit=fraction_fit,
                fraction_eval=fraction_fit,
                eta=eta,
                eta_l=eta_l,
                tau=tau,
                beta_1=beta_1,
                beta_2=beta_2,
                initial_parameters=initial_parameters,
                accept_failures=accept_failures,
                min_fit_clients=min_fit_clients,
                min_eval_clients=min_eval_clients,
                **kwargs
            )
        ),
        accept_failures=accept_failures
    )

    return strategy
