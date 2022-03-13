import functools
from pathlib import Path

import experiments.setup_system_paths as ssp
from create_lda_cifar_datasets import DEFAULT_CONCENTRATIONS

ssp.setup_system_paths()

from experiments.cifar10_experiments.cifar10_metadata_providers import \
    CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER, VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import Cifar10LdaClientDatasetProcessor
from sources.experiments.grid_search_metadata_provider import ParameterGridMetadataGenerator
from sources.models.cifar10_lda.cifar10_lda_model_template import Cifar10LdaModelTemplate

from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation import \
    create_central_evaluation_function_from_dataset_processor
from sources.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    for fraction in DEFAULT_CONCENTRATIONS:
        model_template = Cifar10LdaModelTemplate(DEFAULT_SEED)
        dataset_factory = Cifar10LdaClientDatasetFactory(root_data_dir, 100, fraction)
        total_clients = dataset_factory.get_number_of_clients()
        central_dataset = get_default_iid_dataset("cifar10")

        eval_fn = create_central_evaluation_function_from_dataset_processor(
            model_template,
            central_dataset,
            Cifar10LdaClientDatasetProcessor())

        strategy_provider = functools.partial(
            full_eval_fed_avg_strategy_provider,
            eval_fn
        )

        pgmg = ParameterGridMetadataGenerator(VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP,
                                              lambda d: strategy_provider,
                                              lambda d: model_template.get_optimizer(),
                                              CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER,
                                              lambda d: f"_c_{d['clients_per_round']}")
        pgr = pgmg.generate_grid_responses()

        SimulateExperiment.start_experiment(
            f"Cifar10Lda_{fraction}_Varying_Local_Epochs",
            model_template,
            dataset_factory,
            strategy_provider=None,
            strategy_provider_list=pgr.strategy_provider_list,
            optimizer_list=pgr.optimizer_list,
            experiment_metadata_list=pgr.experiment_metadata_list,
            base_dir=base_dir,
            runs_per_experiment=1,
            centralised_evaluation=True,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=10)
