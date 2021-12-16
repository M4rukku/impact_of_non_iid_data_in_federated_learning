import dataclasses
import json
import logging

import flwr
import tensorflow as tf
from pathlib import Path
from typing import List, Callable

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.experiments.experiment_metadata import ExperimentMetadata, \
    get_simulation_parameters_from_experiment_metadata
from sources.experiments.extended_experiment_metadata import create_extended_experiment_metadata, \
    ExtendedExperimentMetadata
from sources.flwr_parameters.exception_definitions import \
    ExperimentParameterListsHaveUnequalLengths
from sources.flwr_parameters.simulation_parameters import RayInitArgs, ClientResources, \
    DEFAULT_RAY_INIT_ARGS
from sources.flwr_strategies.evaluation_metrics_logging_strategy_decorator import \
    EvaluationMetricsLoggingStrategyDecorator
from sources.flwr_strategies.model_logging_strategy_decorator import \
    ModelLoggingStrategyDecorator
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate
from sources.simulation_framework.ray_based_simulator import RayBasedSimulator


def create_dirname_from_extended_metadata(experiment_metadata: ExtendedExperimentMetadata,
                                          exp: int):
    if "learning_rate" in experiment_metadata.optimizer_config:
        return (
                f"{experiment_metadata.strategy_name}" +
                f"_{experiment_metadata.optimizer_config['name']}" +
                f"_lr{experiment_metadata.optimizer_config['learning_rate']}" +
                f"_nr{experiment_metadata.num_rounds}" +
                f"_nc{experiment_metadata.num_clients}" +
                f"_le{experiment_metadata.local_epochs}_i{exp}"
        )
    else:
        return (
                f"{experiment_metadata.strategy_name}" +
                f"_{experiment_metadata.optimizer_config['name']}" +
                f"_nr{experiment_metadata.num_rounds}" +
                f"_nc{experiment_metadata.num_clients}" +
                f"_le{experiment_metadata.local_epochs}_i{exp}"
        )


class SimulationExperiment:

    @staticmethod
    def _ensure_strategy_implements_basic_config_funcs(
            strategy: flwr.server.strategy.Strategy,
            extended_metadata: ExtendedExperimentMetadata,
    ):
        # Setup fit/evaluate config functions

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
            model_template: ModelTemplate,
            dataset_factory: ClientDatasetFactory,
            strategy_provider: Callable[[ExperimentMetadata], flwr.server.strategy.Strategy],
            experiment_metadata_list: List[ExperimentMetadata],

            base_dir: Path,
            ray_init_args: RayInitArgs = DEFAULT_RAY_INIT_ARGS,
            client_resources: ClientResources = None,

            strategies_list: List[flwr.server.strategy.Strategy] = None,
            optimizer_list: List[tf.keras.optimizers.Optimizer] = None,
            fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
            evaluation_callbacks: list[tf.keras.callbacks.Callback] = None,
            metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
    ):

        strategies_list_defined = True if strategies_list is not None else False
        optimizer_list_defined = True if optimizer_list is not None else False
        length = len(experiment_metadata_list)

        # Check whether lengths are equivalent
        if strategies_list_defined and len(strategies_list) != length:
            raise ExperimentParameterListsHaveUnequalLengths()
        if optimizer_list_defined and len(optimizer_list) != length:
            raise ExperimentParameterListsHaveUnequalLengths()

        # Start Simulations
        checkpoint_dir = base_dir / "checkpoints"
        experiment_dir = checkpoint_dir / experiment_name

        logging.info(f"Starting to execute simulations.")

        for i in range(length):
            logging.info(f"Starting to execute simulation round {i + 1}.")

            # Setup base experiment/strategy/optimizer data
            experiment_metadata = experiment_metadata_list[i]
            strategy_ = strategy_provider(experiment_metadata)
            if strategies_list_defined:
                strategy_ = strategies_list[i]
            if optimizer_list_defined:
                model_template.set_optimizer(optimizer_list[i])

            strategy_name = type(strategy_).__name__
            extended_metadata = create_extended_experiment_metadata(experiment_metadata,
                                                                    strategy_name,
                                                                    model_template.get_optimizer())

            # Setup Directory Structure
            dirname = create_dirname_from_extended_metadata(extended_metadata, i)
            inner_experiment_dir = experiment_dir / dirname
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
                json.dump(extended_metadata_dict, f)

            # Setup fit/evaluate config functions
            SimulationExperiment._ensure_strategy_implements_basic_config_funcs(strategy_,
                                                                                extended_metadata)

            # Add decorators for logging
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

            # Start Simulation
            simulator = RayBasedSimulator(
                get_simulation_parameters_from_experiment_metadata(experiment_metadata),
                strategy_,
                model_template,
                dataset_factory,
                fitting_callbacks,
                evaluation_callbacks,
                metrics,
                client_resources=client_resources,
                ray_init_args=ray_init_args)

            simulator.start_simulation()

            logging.info(f"Finished executing simulation round {i + 1}.")
