import os
import flwr as fl
from pathlib import Path

from sources.datasets.femnist.femnist_client_dataset_factory import FemnistClientDatasetFactory
from sources.flwr_parameters.default_parameters import DEFAULT_SEED
from sources.flwr_strategies.evaluation_metrics_logging_strategy_decorator import \
    EvaluationMetricsLoggingStrategyDecorator
from sources.flwr_strategies.model_logging_strategy_decorator import ModelLoggingStrategyDecorator
from sources.models.femnist.femnist_model_template import FemnistModelTemplate
from sources.simulation_framework.base_simulator import BaseSimulator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    base_dir = Path(__file__).parent.parent.parent
    checkpoint_dir = base_dir / "checkpoints"
    data_dir = base_dir / "data"
    experiment_name = "initial_experiment"
    model_saving_dir = str(checkpoint_dir / experiment_name / "models")
    metrics_saving_dir = str(checkpoint_dir / experiment_name / "metrics")


    def eval_config(rnd: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": 32,
            "val_steps": 2
        }
        return config


    def fit_config(rnd: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": 32,
            "local_epochs": 1 if rnd < 2 else 2,
        }
        return config


    simulation_parameters = {"num_rounds": 2, "num_clients": 100}

    strategy = EvaluationMetricsLoggingStrategyDecorator(
        strategy=fl.server.strategy.FedAvg(on_fit_config_fn=fit_config, on_evaluate_config_fn=eval_config),
        metrics_logging_folder=metrics_saving_dir,
        experiment_identifier=experiment_name
    )
    strategy = ModelLoggingStrategyDecorator(
        strategy=strategy,
        model_saving_folder=model_saving_dir,
        experiment_identifier=experiment_name
    )

    model_template = FemnistModelTemplate(DEFAULT_SEED)
    dataset_factory = FemnistClientDatasetFactory(data_dir)

    simulator = BaseSimulator(simulation_parameters, strategy, model_template, dataset_factory)

    simulator.start_simulation()
