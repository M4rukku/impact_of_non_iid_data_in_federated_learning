import functools
from pathlib import Path

from experiments.verification_experiments.initial_experiment_metadata_providers import \
    shakespeare_initial_experiment_metadata_provider, \
    shakespeare_initial_experiment_optimizer_provider
from sources.datasets.shakespeare.shakespeare_client_dataset_factory import \
    ShakespeareClientDatasetFactory
from sources.models.shakespeare.shakespeare_model_template import ShakespeareKerasModelTemplate
from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.datasets.shakespeare.shakespeare_client_dataset_processor import \
    ShakespeareClientDatasetProcessor
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = ShakespeareKerasModelTemplate(DEFAULT_SEED)
    dataset_factory = ShakespeareClientDatasetFactory(root_data_dir)
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("shakespeare")

    eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
        model_template,
        central_dataset,
        ShakespeareClientDatasetProcessor())

    strategy_provider = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn
    )

    experiment_metadata_list = shakespeare_initial_experiment_metadata_provider(total_clients)
    optimizer_list = shakespeare_initial_experiment_optimizer_provider()

    SimulateExperiment.start_experiment(
        "InitialShakespeareExperiments",
        model_template,
        dataset_factory,
        strategy_provider,
        experiment_metadata_list,
        base_dir,
        optimizer_list=optimizer_list,
        runs_per_experiment=2,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=2)
