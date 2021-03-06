import functools
from pathlib import Path

from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name

from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10
from experiments.cifar10_experiments.cifar10_metadata_providers import \
    VARYING_REPORTING_FRACTION_EXP_PARAMETER_MAP, \
    CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import \
    Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import \
    Cifar10LdaClientDatasetProcessor
from sources.experiments.grid_search_metadata_provider import ParameterGridMetadataGenerator
from sources.models.cifar10_lda.cifar10_lda_model_template import Cifar10LdaKerasModelTemplate

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED


def vrf_cifar10():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    for concentration in DEFAULT_CONCENTRATIONS_CIFAR10:
        model_template = Cifar10LdaKerasModelTemplate(DEFAULT_SEED)
        dataset_factory = Cifar10LdaClientDatasetFactory(root_data_dir, 100, concentration)
        central_dataset = get_default_iid_dataset(get_lda_cifar10_dataset_name(concentration, 100))

        eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
            model_template,
            central_dataset,
            Cifar10LdaClientDatasetProcessor())

        strategy_provider = functools.partial(
            full_eval_fed_avg_strategy_provider,
            eval_fn
        )

        pgmg = ParameterGridMetadataGenerator(VARYING_REPORTING_FRACTION_EXP_PARAMETER_MAP,
                                              lambda d: strategy_provider,
                                              lambda d: model_template.get_optimizer(),
                                              CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER,
                                              lambda d: f"_c_{d['clients_per_round']}")
        pgr = pgmg.generate_grid_responses()

        SimulateExperiment.start_experiment(
            f"Cifar10Lda_{concentration}_Varying_Clients_Per_Round",
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
