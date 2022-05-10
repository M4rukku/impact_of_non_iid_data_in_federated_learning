import functools
import math
from pathlib import Path

from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator

from experiments.shakespeare_experiments.shakespeare_metadata_providers import \
    SHAKESPEARE_BASE_METADATA_HYPERPARAMETER_SEARCH_PROVIDER
from sources.datasets.shakespeare.shakespeare_client_dataset_factory import \
    ShakespeareClientDatasetFactory
from sources.datasets.shakespeare.shakespeare_client_dataset_processor import \
    ShakespeareClientDatasetProcessor
from sources.models.shakespeare.shakespeare_model_template import ShakespeareKerasModelTemplate

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED


def e(exp):
    return math.pow(10, exp)


def shakespeare_fedavg():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = ShakespeareKerasModelTemplate(DEFAULT_SEED)
    dataset_factory = ShakespeareClientDatasetFactory(root_data_dir)
    central_dataset = get_default_iid_dataset("shakespeare")

    eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
        model_template,
        central_dataset,
        ShakespeareClientDatasetProcessor())

    initial_model = model_template.get_model()
    initial_parameters = initial_model.get_weights()

    local_learning_rates = [e(-3.0), e(-2.0), e(-1.0), e(0.0)]

    fed_avg = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn,
        initial_parameters=initial_parameters
    )

    SimulateExperiment.start_experiment(
        f"Shakespeare_Fedavg",
        model_template,
        dataset_factory,
        strategy_provider=None,
        strategy_provider_list=[fed_avg for _ in local_learning_rates],
        optimizer_list=[model_template.get_optimizer(lr) for lr in local_learning_rates],
        experiment_metadata_list=[SHAKESPEARE_BASE_METADATA_HYPERPARAMETER_SEARCH_PROVIDER() for
                                  _ in local_learning_rates],
        base_dir=base_dir,
        runs_per_experiment=2,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=2,
        simulator_provider=SerialExecutionSimulator
    )

