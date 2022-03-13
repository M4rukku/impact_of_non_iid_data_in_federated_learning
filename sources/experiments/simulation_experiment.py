import csv
import dataclasses
import json
import logging
import pickle
from collections import defaultdict

import flwr
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Callable, Dict, Union, Optional

from flwr.server import SimpleClientManager
from flwr.server.strategy import Strategy

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import ClientDatasetFactory
from sources.experiments.experiment_metadata import ExperimentMetadata, \
    get_simulation_parameters_from_experiment_metadata
from sources.experiments.extended_experiment_metadata import create_extended_experiment_metadata, \
    ExtendedExperimentMetadata
from sources.flwr_parameters.exception_definitions import \
    ExperimentParameterListsHaveUnequalLengths, NoStrategyProviderError
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED, set_global_determinism
from sources.flwr_parameters.simulation_parameters import RayInitArgs, ClientResources, \
    DEFAULT_RAY_INIT_ARGS, DEFAULT_RUNS_PER_EXPERIMENT, EarlyStoppingSimulationParameters
from sources.flwr_strategies_decorators.base_strategy_decorator import get_name_of_strategy
from sources.flwr_strategies_decorators.central_evaluation_logging_decorator import \
    CentralEvaluationLoggingDecorator
from sources.flwr_strategies_decorators.evaluation_metrics_logging_strategy_decorator import \
    EvaluationMetricsLoggingStrategyDecorator
from sources.flwr_strategies_decorators.model_logging_strategy_decorator import \
    ModelLoggingStrategyDecorator
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate
from sources.simulation_framework.early_stopping_server import EarlyStoppingServer
from sources.simulation_framework.ray_based_simulator import RayBasedSimulator


def create_dirname_from_extended_metadata(experiment_metadata: ExtendedExperimentMetadata,
                                          exp: int):
    custom_suffix = experiment_metadata.custom_suffix \
        if experiment_metadata.custom_suffix is not None else ""

    if "learning_rate" in experiment_metadata.optimizer_config:
        return (
                f"{experiment_metadata.strategy_name}" +
                f"_{experiment_metadata.optimizer_config['name']}" +
                f"_lr{experiment_metadata.optimizer_config['learning_rate']}" +
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


def average_experiment_runs(base_experiment_dir):
    experiment_rounds = base_experiment_dir.iterdir()
    initial_experiment = next(experiment_rounds)
    experiment_rounds = list(base_experiment_dir.iterdir())
    experiment_rounds.sort()

    pkl_files = list(base_experiment_dir.glob("*/metrics/*.pkl"))

    def load_pkl(path):
        with path.open("rb") as f:
            data = pickle.load(f)
        return data

    loaded_pkl_files = defaultdict(list)
    for file_path in pkl_files:
        loaded_pkl_files[file_path.stem].append(load_pkl(file_path))

    def avg_l_dicts(l_dict: List[Dict[str, Union[float, int]]]):
        epoch_result_dict = defaultdict(lambda: 0.0)

        # Sum all dict results from that epoch
        num_dicts = 0
        for dict_ in l_dict:
            if not any(pd.isna(list(dict_.values()))):
                for key, val in dict_.items():
                    epoch_result_dict[key] += val
                num_dicts += 1
            else:
                pass

        return {key: val / num_dicts for key, val in epoch_result_dict.items()}

    avg_pkl_data = {filename: avg_l_dicts(same_epoch_files) for filename, same_epoch_files in
                    loaded_pkl_files.items()}

    experiment_name = base_experiment_dir.name
    avg_eval_metrics = base_experiment_dir / f"avg_evaluation_metrics_{experiment_name}.pkl"
    with avg_eval_metrics.open("wb") as f:
        pickle.dump(avg_pkl_data, f)

    # Avg CSV Files
    csv_files = list(base_experiment_dir.glob("*/metrics/*.csv"))

    def load_csv(csv_file: Path):
        with csv_file.open(newline="") as f:
            data = list(map(float, *csv.reader(f)))
        return data

    loaded_csv_data_arrays = []
    for file_path in csv_files:
        loaded_csv_data_arrays.append(load_csv(file_path))  # array of arrays

    round_datapoints_map = defaultdict(list)
    for data_array in loaded_csv_data_arrays:
        for i, data_point in enumerate(data_array):
            round_datapoints_map[i].append(data_array[i])

    # Note that the dict is ordered by insertion order = index order
    avg_accuracy_data = np.array(list(map(lambda ls: np.average(np.array(ls)), round_datapoints_map.values())))

    avg_accuracy_file = base_experiment_dir / "avg_accuracy_metrics.csv"
    with avg_accuracy_file.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(avg_accuracy_data)

    (base_experiment_dir / "experiment_metadata_file.json").write_text(
        (initial_experiment / "experiment_metadata_file.json").read_text())


class SimulationExperiment:

    @staticmethod
    def _ensure_strategy_implements_basic_config_funcs(
            strategy: flwr.server.strategy.Strategy,
            extended_metadata: ExtendedExperimentMetadata,
            centralised_evaluation: bool,
            aggregated_evaluation: bool
    ):
        # Setup fit/evaluate config functions
        if hasattr(strategy, "strategy"):
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
            model_template: ModelTemplate,
            dataset_factory: ClientDatasetFactory,
            strategy_provider: Optional[Callable[[ExperimentMetadata], Strategy]],
            experiment_metadata_list: List[ExperimentMetadata],

            base_dir: Path,
            ray_init_args: RayInitArgs = DEFAULT_RAY_INIT_ARGS,
            client_resources: ClientResources = None,

            strategy_provider_list: Optional[List[Callable[[ExperimentMetadata], Strategy]]] = None,
            optimizer_list: List[tf.keras.optimizers.Optimizer] = None,
            fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
            evaluation_callbacks: list[tf.keras.callbacks.Callback] = None,
            metrics: list[tf.keras.metrics.Metric] = DEFAULT_METRICS,
            seed: int = DEFAULT_SEED,
            runs_per_experiment: int = DEFAULT_RUNS_PER_EXPERIMENT,
            centralised_evaluation=False,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=10
    ):

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
                extended_metadata = create_extended_experiment_metadata(experiment_metadata,
                                                                        strategy_name,
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
                    json.dump(extended_metadata_dict, f)

                # Setup fit/evaluate config functions
                SimulationExperiment._ensure_strategy_implements_basic_config_funcs(
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
                simulation_parameters = get_simulation_parameters_from_experiment_metadata(experiment_metadata)
                server = None
                if isinstance(simulation_parameters, EarlyStoppingSimulationParameters):
                    server = EarlyStoppingServer(SimpleClientManager(), strategy_)

                simulator = RayBasedSimulator(
                    simulation_parameters,
                    strategy_,
                    model_template,
                    dataset_factory,
                    fitting_callbacks,
                    evaluation_callbacks,
                    metrics,
                    client_resources=client_resources,
                    ray_init_args=ray_init_args,
                    server=server)

                simulator.start_simulation()
                logging.info(f"Finished run {run + 1}/{runs_per_experiment} of experiment {i + 1}.")

            logging.info(f"Averaging runs for experiment {i + 1}.")

            average_experiment_runs(base_experiment_dir)

            logging.info(f"Finished executing simulation round {i + 1}.")
