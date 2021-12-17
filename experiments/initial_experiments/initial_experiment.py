import os
import flwr as fl
from pathlib import Path
# Add
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from sources.datasets.femnist.femnist_client_dataset_factory import \
    FemnistClientDatasetFactory
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.flwr_parameters.default_strategy_configs import \
    femnist_eval_config, femnist_fit_config
from sources.flwr_strategies.evaluation_metrics_logging_strategy_decorator \
    import EvaluationMetricsLoggingStrategyDecorator
from sources.flwr_strategies.model_logging_strategy_decorator import \
    ModelLoggingStrategyDecorator
from sources.models.femnist.femnist_model_template import FemnistModelTemplate
from sources.simulation_framework.ray_based_simulator import RayBasedSimulator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    checkpoint_dir = base_dir / "checkpoints"
    data_dir = base_dir / "data"
    experiment_name = "initial_experiment"
    model_saving_dir = str(checkpoint_dir / experiment_name / "models")
    metrics_saving_dir = str(checkpoint_dir / experiment_name / "metrics")

    simulation_parameters = {"num_rounds": 2, "num_clients": 2}
    strategy = \
        fl.server.strategy.FedAvg(on_fit_config_fn=femnist_fit_config,
                                  on_evaluate_config_fn=femnist_eval_config)

    strategy = EvaluationMetricsLoggingStrategyDecorator(
        strategy=strategy,
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

    simulator = RayBasedSimulator(simulation_parameters,
                                  strategy,
                                  model_template,
                                  dataset_factory)

    simulator.start_simulation()
