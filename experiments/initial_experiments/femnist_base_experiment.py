from pathlib import Path
import flwr
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from sources.datasets.femnist.femnist_client_dataset_factory import FemnistClientDatasetFactory
from sources.experiments.experiment_metadata import ExperimentMetadata
from sources.experiments.simulation_experiment import SimulationExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.models.femnist.femnist_model_template import FemnistModelTemplate

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"


    def strategy_provider(experiment_metadata: ExperimentMetadata):
        if experiment_metadata.clients_per_round >= 1:
            fraction_fit = float(experiment_metadata.clients_per_round) / \
                           float(experiment_metadata.num_clients)
        else:
            fraction_fit = experiment_metadata.clients_per_round

        return flwr.server.strategy.FedAvg(fraction_fit=fraction_fit, fraction_eval=fraction_fit)


    experiment_metadata_list = [
        ExperimentMetadata(600, 5, 3, 5, 1, 2),
        ExperimentMetadata(600, 5, 3, 5, 1, 2),
        ExperimentMetadata(600, 5, 3, 5, 1, 2),
    ]

    SimulationExperiment.start_experiment(
        "FemnistBaseExperiment7",
        FemnistModelTemplate(DEFAULT_SEED),
        FemnistClientDatasetFactory(str(root_data_dir.resolve())),
        strategy_provider,
        experiment_metadata_list,
        base_dir,
        runs_per_experiment=3)
