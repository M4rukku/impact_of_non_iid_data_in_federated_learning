import os
import sys
from pathlib import Path

from sources.flwr_strategies.FullEvaluationFedAvg import FullEvaluationFedAvg

sys.path.append(str(Path(os.getcwd()).parent.parent.resolve()))
dllpath = Path("C:") / "Program Files" / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.2" / "bin"
dllstring = str(dllpath.resolve())
os.add_dll_directory(dllstring)

from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation import create_central_evaluation_function_from_dataset

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from sources.datasets.femnist.femnist_client_dataset_factory import FemnistClientDatasetFactory
from sources.experiments.experiment_metadata import ExperimentMetadata
from sources.experiments.simulation_experiment import SimulationExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED, set_global_determinism
from sources.models.femnist.femnist_model_template import FemnistModelTemplate

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = FemnistModelTemplate(DEFAULT_SEED)
    dataset_factory = FemnistClientDatasetFactory(str(root_data_dir.resolve()))
    central_dataset = get_default_iid_dataset("femnist")
    dataset = dataset_factory.create_dataset("1")
    eval_fn = create_central_evaluation_function_from_dataset(model_template,
                                                              central_dataset,
                                                              dataset)


    def strategy_provider(experiment_metadata: ExperimentMetadata):
        if experiment_metadata.clients_per_round >= 1:
            fraction_fit = float(experiment_metadata.clients_per_round) / \
                           float(experiment_metadata.num_clients)
        else:
            fraction_fit = experiment_metadata.clients_per_round

        strategy = FullEvaluationFedAvg(eval_fn=eval_fn,
                                        fraction_fit=fraction_fit,
                                        fraction_eval=fraction_fit)

        return strategy


    experiment_metadata_list = [
        ExperimentMetadata(num_clients=2500,
                           num_rounds=100,
                           clients_per_round=10,
                           batch_size=10,
                           local_epochs=1,
                           val_steps=3),
    ]

    SimulationExperiment.start_experiment(
        "FullEvaluationFemnist",
        model_template,
        dataset_factory,
        strategy_provider,
        experiment_metadata_list,
        base_dir,
        runs_per_experiment=1,
        centralised_evaluation=True,
        aggregated_evaluation=True)
