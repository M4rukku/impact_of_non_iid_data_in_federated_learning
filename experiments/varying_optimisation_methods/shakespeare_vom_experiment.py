import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
sys.path.append(str(Path(os.getcwd()).parent.parent.resolve()))

dllpath = Path("C:") / "Program Files" / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.2" / "bin"
if dllpath.exists():
    dllstring = str(dllpath.resolve())
    os.add_dll_directory(dllstring)

from experiments.varying_optimisation_methods.vom_experiment_metadata_providers import \
    vom_experiments_strategy_providers, shakespeare_vom_experiment_metadata_provider, \
    shakespeare_vom_experiment_optimizer_provider
from sources.datasets.shakespeare_iid.shakespeare_iid_client_dataset_factory import \
    ShakespeareIIDClientDatasetFactory
from sources.models.shakespeare.shakespeare_model_template import ShakespeareModelTemplate
from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation import create_central_evaluation_function_from_dataset

from sources.experiments.simulation_experiment import SimulationExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED, set_seeds

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    set_seeds(DEFAULT_SEED)
    model_template = ShakespeareModelTemplate(DEFAULT_SEED)
    initial_parameters = model_template.get_model().get_weights()

    dataset_factory = ShakespeareIIDClientDatasetFactory(str(root_data_dir.resolve()))
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("shakespeare")
    dataset = dataset_factory.create_dataset("1")
    eval_fn = create_central_evaluation_function_from_dataset(model_template,
                                                              central_dataset,
                                                              dataset)

    strategy_providers = vom_experiments_strategy_providers(eval_fn)
    experiment_metadata_list = shakespeare_vom_experiment_metadata_provider(total_clients)
    optimizer_list = shakespeare_vom_experiment_optimizer_provider()

    SimulationExperiment.start_experiment(
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
