import functools
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
sys.path.append(str(Path(os.getcwd()).parent.parent.resolve()))

dllpath = Path("C:") / "Program Files" / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.2" / "bin"
if dllpath.exists():
    dllstring = str(dllpath.resolve())
    os.add_dll_directory(dllstring)

from experiments.iid_data_initial_experiments.initial_iid_experiment_metadata_providers import \
    celeba_initial_iid_experiment_metadata_provider, celeba_initial_iid_experiment_optimizer_provider
from sources.datasets.celeba_iid.celeba_iid_client_dataset_factory import \
    CelebaIIDClientDatasetFactory
from sources.models.celeba.celeba_model_template import CelebaModelTemplate
from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation import create_central_evaluation_function_from_dataset
from sources.flwr_strategies.full_evaluation_strategy_providers import full_eval_fed_avg_strategy_provider

from sources.experiments.simulation_experiment import SimulationExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = CelebaModelTemplate(DEFAULT_SEED)
    dataset_factory = CelebaIIDClientDatasetFactory(str(root_data_dir.resolve()))
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("celeba")
    dataset = dataset_factory.create_dataset("1")
    eval_fn = create_central_evaluation_function_from_dataset(model_template,
                                                              central_dataset,
                                                              dataset)
    strategy_provider = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn
    )
    experiment_metadata_list = celeba_initial_iid_experiment_metadata_provider(total_clients)
    optimizer_list = celeba_initial_iid_experiment_optimizer_provider()

    SimulationExperiment.start_experiment(
        "InitialCelebaIIDExperiment_",
        model_template,
        dataset_factory,
        strategy_provider,
        experiment_metadata_list,
        base_dir,
        runs_per_experiment=3,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=10)
