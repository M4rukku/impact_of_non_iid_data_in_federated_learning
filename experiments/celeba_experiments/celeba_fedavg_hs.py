import functools
import math
from pathlib import Path

from experiments.celeba_experiments.celeba_metadata_providers import \
    CELEBA_BASE_METADATA_OPT_EXP_PROVIDER, CELEBA_BASE_METADATA_HYPERPARAMETER_SEARCH_PROVIDER
from sources.datasets.celeba.celeba_client_dataset_factory import CelebaClientDatasetFactory
from sources.datasets.celeba.celeba_client_dataset_processor import CelebaClientDatasetProcessor
from sources.models.celeba.celeba_model_template import CelebaKerasModelTemplate

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator


def e(exp):
    return math.pow(10, exp)


def celeba_fedavg():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = CelebaKerasModelTemplate(DEFAULT_SEED)
    dataset_factory = CelebaClientDatasetFactory(root_data_dir)
    central_dataset = get_default_iid_dataset("celeba")

    eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
        model_template,
        central_dataset,
        CelebaClientDatasetProcessor())

    initial_model = model_template.get_model()
    initial_parameters = initial_model.get_weights()

    local_learning_rates = [e(-3.0), e(-2.0), e(-1.0), e(0.0)]

    fed_avg = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn,
        initial_parameters=initial_parameters
    )

    SimulateExperiment.start_experiment(
        f"Celeba_Fedavg",
        model_template,
        dataset_factory,
        strategy_provider=None,
        strategy_provider_list=[fed_avg for _ in local_learning_rates],
        optimizer_list=[model_template.get_optimizer(lr) for lr in local_learning_rates],
        experiment_metadata_list=[
            CELEBA_BASE_METADATA_HYPERPARAMETER_SEARCH_PROVIDER()
            for _ in local_learning_rates],
        base_dir=base_dir,
        runs_per_experiment=5,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=10,
        simulator_provider=SerialExecutionSimulator,
        simulator_args={})
