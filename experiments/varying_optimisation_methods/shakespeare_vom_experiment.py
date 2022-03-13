from pathlib import Path

import experiments.setup_system_paths as ssp

ssp.setup_system_paths()

from sources.datasets.shakespeare.shakespeare_client_dataset_factory import \
    ShakespeareClientDatasetFactory
from experiments.varying_optimisation_methods.vom_experiment_metadata_providers import \
    vom_experiments_strategy_providers, shakespeare_vom_experiment_metadata_provider, \
    shakespeare_vom_experiment_optimizer_provider
from sources.datasets.shakespeare.shakespeare_client_dataset_processor import \
    ShakespeareClientDatasetProcessor
from sources.models.shakespeare.shakespeare_model_template import ShakespeareModelTemplate
from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation import \
    create_central_evaluation_function_from_dataset_processor

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED, set_seeds

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    set_seeds(DEFAULT_SEED)
    model_template = ShakespeareModelTemplate(DEFAULT_SEED)
    initial_parameters = model_template.get_model().get_weights()

    dataset_factory = ShakespeareClientDatasetFactory(root_data_dir)
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("shakespeare")

    eval_fn = create_central_evaluation_function_from_dataset_processor(
        model_template,
        central_dataset,
        ShakespeareClientDatasetProcessor())

    strategy_providers = vom_experiments_strategy_providers(eval_fn)
    experiment_metadata_list = shakespeare_vom_experiment_metadata_provider(total_clients)
    optimizer_list = shakespeare_vom_experiment_optimizer_provider()

    SimulateExperiment.start_experiment(
        "Shakespeare_Varying_Optimisers_Experiment",
        model_template,
        dataset_factory,
        None,
        experiment_metadata_list,
        base_dir,
        strategy_provider_list=strategy_providers,
        optimizer_list=optimizer_list,
        runs_per_experiment=1,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=2)
