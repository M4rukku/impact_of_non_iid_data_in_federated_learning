import functools
import math
from pathlib import Path

from experiments.verification_experiments.initial_experiment_metadata_providers import \
    celeba_initial_experiment_metadata_provider, celeba_initial_experiment_optimizer_provider
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

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = CelebaKerasModelTemplate(DEFAULT_SEED)
    dataset_factory = CelebaClientDatasetFactory(root_data_dir)
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("celeba")

    eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
        model_template,
        central_dataset,
        CelebaClientDatasetProcessor())

    strategy_provider = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn
    )

    experiment_metadata_list = celeba_initial_experiment_metadata_provider(
        math.ceil(total_clients * 0.1)
    )
    optimizer_list = celeba_initial_experiment_optimizer_provider()

    SimulateExperiment.start_experiment(
        "InitialCelebaExperiments",
        model_template,
        dataset_factory,
        strategy_provider,
        experiment_metadata_list,
        base_dir,
        runs_per_experiment=3,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=10)
