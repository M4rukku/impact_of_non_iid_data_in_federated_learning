import dataclasses
import json
import logging
import math

import flwr
import tensorflow as tf
from pathlib import Path
from typing import List, Callable, Optional, Type

from flwr.server.strategy import Strategy

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import \
    ClientDatasetFactory
from sources.experiments.average_experiment_runs import average_experiment_runs
from sources.experiments.experiment_metadata import ExperimentMetadata, \
    get_simulation_parameters_from_experiment_metadata
from sources.experiments.extended_experiment_metadata import \
    create_extended_experiment_metadata, \
    ExtendedExperimentMetadata
from sources.models.base_model_template import BaseModelTemplate
from sources.utils.exception_definitions import \
    ExperimentParameterListsHaveUnequalLengths, NoStrategyProviderError
from sources.utils.set_random_seeds import DEFAULT_SEED, set_global_determinism
from sources.utils.simulation_parameters import DEFAULT_RUNS_PER_EXPERIMENT
from sources.flwr.flwr_strategies_decorators.base_strategy_decorator import get_name_of_strategy
from sources.flwr.flwr_strategies_decorators.central_evaluation_logging_decorator import \
    CentralEvaluationLoggingDecorator
from sources.flwr.flwr_strategies_decorators.evaluation_metrics_logging_strategy_decorator import \
    EvaluationMetricsLoggingStrategyDecorator
from sources.flwr.flwr_strategies_decorators.model_logging_strategy_decorator import \
    ModelLoggingStrategyDecorator
from sources.metrics.default_metrics_tf import DEFAULT_METRICS
from sources.simulators.base_simulator import BaseSimulator
from sources.simulators.ray_based_simulator.ray_based_simulator import RayBasedSimulator, \
    default_ray_args


def round_to_two_nonzero_digits(n):
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-math.floor(math.log10(abs(n))))
    if scale <= 0:
        scale = 1
    factor = 10 ** (scale + 1)
    return sgn * math.floor(abs(n) * factor) / factor


def create_dirname_from_extended_metadata(experiment_metadata: ExtendedExperimentMetadata,
                                          exp: int):
    custom_suffix = experiment_metadata.custom_suffix \
        if experiment_metadata.custom_suffix is not None else ""

    if experiment_metadata.local_learning_rate is not None:
        return (
                f"{experiment_metadata.strategy_name}" +
                f"_{experiment_metadata.optimizer_config['name']}" +
                f"_lr{round_to_two_nonzero_digits(experiment_metadata.optimizer_config['learning_rate'])}" +
                f"_nr{experiment_metadata.num_rounds}" +
                f"_nc{experiment_metadata.num_clients}" +
                f"_le{experiment_metadata.local_epochs}_i{exp}" +
                custom_suffix
        )
    else:
        return (
                f"{experiment_metadata.strategy_name}" +
                f"_{experiment_metadata.optimizer_config['name']}" +
                f"_nr{experiment_metadata.num_rounds}" +
                f"_nc{experiment_metadata.num_clients}" +
                f"_le{experiment_metadata.local_epochs}_i{exp}" +
                custom_suffix
        )


class SimulateExperiment:

    @staticmethod
    def _ensure_strategy_implements_basic_config_funcs(
            strategy: flwr.server.strategy.Strategy,
            extended_metadata: ExtendedExperimentMetadata,
            centralised_evaluation: bool,
            aggregated_evaluation: bool
    ):
        # Setup fit/evaluate config functions
        while hasattr(strategy, "strategy"):
            strategy = strategy.strategy

        def on_fit_config_fn(rnd: int):
            config = {
                "batch_size": extended_metadata.batch_size,
                "local_epochs": extended_metadata.local_epochs,
            }
            return config

        def on_evaluate_config_fn(rnd: int):
            config = {
                "batch_size": extended_metadata.batch_size,
                "val_steps": extended_metadata.val_steps
            }
            return config

        if strategy.on_fit_config_fn is None:
            strategy.on_fit_config_fn = on_fit_config_fn
        else:
            fit_fun = strategy.on_fit_config_fn

            def decorated_on_fit_config_fun(rnd: int):
                result_dict = fit_fun(rnd)
                result_dict["batch_size"] = result_dict.get("batch_size",
                                                            extended_metadata.batch_size)
                result_dict["local_epochs"] = result_dict.get("local_epochs",
                                                              extended_metadata.local_epochs)
                return result_dict

            strategy.on_fit_config_fn = decorated_on_fit_config_fun

        if aggregated_evaluation:
            if strategy.on_evaluate_config_fn is None:
                strategy.on_evaluate_config_fn = on_evaluate_config_fn
            else:
                eval_fun = strategy.on_evaluate_config_fn

                def decorated_on_eval_config_fun(rnd: int):
                    result_dict = eval_fun(rnd)
                    result_dict["batch_size"] = result_dict.get("batch_size",
                                                                extended_metadata.batch_size)
                    result_dict["val_steps"] = result_dict.get("val_steps",
                                                               extended_metadata.val_steps)
                    return result_dict

                strategy.on_evaluate_config_fn = decorated_on_eval_config_fun

    @staticmethod
    def start_experiment(
            experiment_name: str,
            model_template: BaseModelTemplate,
            dataset_factory: ClientDatasetFactory,
            base_dir: Path,
            experiment_metadata_list: List[ExperimentMetadata],
            strategy_provider_list: Optional[List[Callable[[ExperimentMetadata], Strategy]]] = None,
            strategy_provider: Optional[Callable[[ExperimentMetadata], Strategy]] = None,
            optimizer_list: List[tf.keras.optimizers.Optimizer] = None,
            metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
            seed: int = DEFAULT_SEED,

            runs_per_experiment: int = DEFAULT_RUNS_PER_EXPERIMENT,
            centralised_evaluation=False,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=10,
            simulator_provider: Type[BaseSimulator] = RayBasedSimulator,
            simulator_args=None,
            **kwargs
    ):
        if simulator_args is None:
            simulator_args = default_ray_args

        logging.basicConfig(level=logging.INFO)
        strategies_list_defined = True if strategy_provider_list is not None else False
        optimizer_list_defined = True if optimizer_list is not None else False
        length = len(experiment_metadata_list)

        # Check whether lengths are equivalent
        if strategies_list_defined and len(strategy_provider_list) != length:
            raise ExperimentParameterListsHaveUnequalLengths()
        if optimizer_list_defined and len(optimizer_list) != length:
            raise ExperimentParameterListsHaveUnequalLengths()

        # Start Simulations
        checkpoint_dir = base_dir / "checkpoints"
        experiment_dir = checkpoint_dir / experiment_name

        logging.info(f"Starting to execute simulations.")

        for i in range(length):
            logging.info(f"Starting to execute simulation round {i + 1}.")
            tf.keras.backend.clear_session()
            set_global_determinism(seed)

            for run in range(runs_per_experiment):
                logging.info(
                    f"Executing run {run + 1}/{runs_per_experiment} of experiment {i + 1}.")
                # Setup base experiment/strategy/optimizer data
                experiment_metadata = experiment_metadata_list[i]

                if strategy_provider is not None:
                    logging.info("Using Global Strategy for Providing Strategies")
                    strategy_ = strategy_provider(experiment_metadata)
                elif strategies_list_defined:
                    logging.info("Using Strategy List for providing strategies")
                    strategy_ = (strategy_provider_list[i])(experiment_metadata=experiment_metadata)
                else:
                    raise NoStrategyProviderError("No Strategy Provider Defined")

                if optimizer_list_defined:
                    model_template.set_optimizer(optimizer_list[i])

                strategy_name = get_name_of_strategy(strategy_)
                extended_metadata = \
                    create_extended_experiment_metadata(experiment_metadata,
                                                        strategy_name,
                                                        model_template.get_optimizer_config(),
                                                        model_template.get_optimizer())

                # Setup Directory Structure
                dirname = create_dirname_from_extended_metadata(extended_metadata, i)
                base_experiment_dir = experiment_dir / dirname
                inner_experiment_dir = base_experiment_dir / f"{dirname}_run_{run}"
                model_saving_dir = inner_experiment_dir / "models"
                model_saving_dir_str = str(model_saving_dir)
                metrics_saving_dir = inner_experiment_dir / "metrics"
                metrics_saving_dir_str = str(metrics_saving_dir)

                inner_experiment_dir.mkdir(parents=True, exist_ok=False)
                model_saving_dir.mkdir(parents=True, exist_ok=False)
                metrics_saving_dir.mkdir(parents=True, exist_ok=False)

                # Store Experiment Metadata
                extended_metadata_dict = dataclasses.asdict(extended_metadata)
                extended_metadata_file = inner_experiment_dir / "experiment_metadata_file.json"

                with extended_metadata_file.open("w") as f:
                    def change_floats_to_str(dictionary):
                        return {k: (str(round(v, 2)) if isinstance(v, float) else v)
                                for (k, v) in extended_metadata_dict.items()}

                    dict_wo_floats = change_floats_to_str(extended_metadata_dict)
                    dict_wo_floats["optimizer_config"] = str(
                        extended_metadata_dict["optimizer_config"])
                    json.dump(dict_wo_floats, f)

                # Setup fit/evaluate config functions
                SimulateExperiment._ensure_strategy_implements_basic_config_funcs(
                    strategy_,
                    extended_metadata,
                    centralised_evaluation,
                    aggregated_evaluation
                )

                # Add decorators for logging
                if aggregated_evaluation:
                    strategy_ = EvaluationMetricsLoggingStrategyDecorator(
                        strategy=strategy_,
                        metrics_logging_folder=metrics_saving_dir_str,
                        experiment_identifier=experiment_name
                    )

                strategy_ = ModelLoggingStrategyDecorator(
                    strategy=strategy_,
                    model_saving_folder=model_saving_dir_str,
                    experiment_identifier=experiment_name
                )

                if centralised_evaluation:
                    strategy_ = CentralEvaluationLoggingDecorator(
                        strategy=strategy_,
                        metrics_logging_folder=metrics_saving_dir_str,
                        experiment_identifier=experiment_name,
                        rounds_between_evaluations=rounds_between_centralised_evaluations
                    )

                # Start Simulation
                simulation_parameters = get_simulation_parameters_from_experiment_metadata(
                    experiment_metadata)
                server = None

                simulator = simulator_provider(
                    simulation_parameters,
                    strategy_,
                    model_template,
                    dataset_factory,
                    metrics=metrics,
                    **simulator_args,
                    **kwargs,
                    server=server)

                simulator.start_simulation()
                logging.info(f"Finished run {run + 1}/{runs_per_experiment} of experiment {i + 1}.")

            logging.info(f"Averaging runs for experiment {i + 1}.")

            average_experiment_runs(base_experiment_dir)

            logging.info(f"Finished executing simulation round {i + 1}.")
