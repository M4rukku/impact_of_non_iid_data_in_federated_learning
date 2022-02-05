from pathlib import Path

import experiments.setup_system_paths as ssp
ssp.setup_system_paths()

from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.datasets.femnist.femnist_client_dataset_processor import FemnistClientDatasetProcessor
from sources.metrics.central_evaluation import \
    create_central_evaluation_function_from_dataset_processor
from experiments.varying_optimisation_methods.vom_experiment_metadata_providers import \
    vom_experiments_strategy_providers, femnist_vom_experiment_metadata_provider, \
    femnist_vom_experiment_optimizer_provider
from sources.datasets.femnist_iid.femnist_iid_client_dataset_factory import \
    FemnistIIDClientDatasetFactory
from sources.experiments.simulation_experiment import SimulationExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED, set_seeds
from sources.models.femnist.femnist_model_template import FemnistModelTemplate

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    set_seeds(DEFAULT_SEED)
    model_template = FemnistModelTemplate(DEFAULT_SEED)
    initial_parameters = model_template.get_model().get_weights()

    dataset_factory = FemnistIIDClientDatasetFactory(str(root_data_dir.resolve()))
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("femnist")

    eval_fn = create_central_evaluation_function_from_dataset_processor(
        model_template,
        central_dataset,
        FemnistClientDatasetProcessor()
    )

    strategy_providers = vom_experiments_strategy_providers(eval_fn, initial_parameters)
    experiment_metadata_list = femnist_vom_experiment_metadata_provider(total_clients)
    optimizer_list = femnist_vom_experiment_optimizer_provider()

    SimulationExperiment.start_experiment(
        "Femnist_Varying_Optimisers_Experiment",
        model_template,
        dataset_factory,
        None,
        experiment_metadata_list,
        base_dir,
        optimizer_list=optimizer_list,
        strategy_provider_list=strategy_providers,
        runs_per_experiment=2,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=5
    )
