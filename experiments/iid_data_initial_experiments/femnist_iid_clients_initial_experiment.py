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

from sources.datasets.femnist.femnist_client_dataset_processor import FemnistClientDatasetProcessor
from experiments.iid_data_initial_experiments.initial_iid_experiment_metadata_providers import \
    femnist_initial_iid_experiment_optimizer_provider, \
    femnist_initial_iid_experiment_metadata_provider
from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation import \
    create_central_evaluation_function_from_dataset_processor

from sources.datasets.femnist_iid.femnist_iid_client_dataset_factory import \
    FemnistIIDClientDatasetFactory
from sources.experiments.simulation_experiment import SimulationExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.models.femnist.femnist_model_template import FemnistModelTemplate
from sources.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = FemnistModelTemplate(DEFAULT_SEED)
    dataset_factory = FemnistIIDClientDatasetFactory(str(root_data_dir.resolve()))
    total_clients = dataset_factory.get_number_of_clients()
    central_dataset = get_default_iid_dataset("femnist")

    eval_fn = create_central_evaluation_function_from_dataset_processor(
        model_template,
        central_dataset,
        FemnistClientDatasetProcessor()
    )

    strategy_provider = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn
    )
    experiment_metadata_list = femnist_initial_iid_experiment_metadata_provider(total_clients)
    optimizer_list = femnist_initial_iid_experiment_optimizer_provider()

    SimulationExperiment.start_experiment(
        "InitialFemnistIIDExperiment",
        model_template,
        dataset_factory,
        strategy_provider,
        experiment_metadata_list,
        base_dir,
        optimizer_list=optimizer_list,
        runs_per_experiment=2,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=5
    )
