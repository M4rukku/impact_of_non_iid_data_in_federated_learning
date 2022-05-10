import functools
import math
from pathlib import Path
from typing import Callable, Dict

from flwr.server.strategy import Strategy

from sources.experiments.experiment_metadata import ExperimentMetadata
from sources.experiments.grid_search_metadata_provider import ParameterGridMetadataGenerator

from experiments.cifar10_experiments.cifar10_metadata_providers import \
    CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER
from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import \
    Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import \
    Cifar10LdaClientDatasetProcessor
from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10
from sources.models.cifar10_lda.cifar10_lda_model_template import Cifar10LdaKerasModelTemplate
from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_adam_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator


def e(exp):
    return math.pow(10, exp)


def cifar10_fedadam_hs():
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

        initial_parameters = model_template.get_model().get_weights()

        variation_map = {
            "client_learning_rate": [e(-3.0), e(-2.0), e(-1.0)],
            "server_learning_rate": [e(-3.0), e(-2.0), e(-1.0)]
        }

        tau = e(-3)

        def provides_strategy_provider_for_fed_adam(d: Dict[str, float]) -> \
                Callable[[ExperimentMetadata], Strategy]:
            return functools.partial(
                full_eval_fed_adam_strategy_provider,
                eval_fn,
                initial_parameters=initial_parameters,
                min_fit_clients=1,
                min_eval_clients=1,
                min_available_clients=1,
                eta=d["server_learning_rate"],
                eta_l=d["client_learning_rate"],
                beta_1=0.9,
                beta_2=0.99,
                tau=tau
            )

        pgmg = ParameterGridMetadataGenerator(variation_map,
                                              provides_strategy_provider_for_fed_adam,
                                              lambda d: model_template.get_optimizer(
                                                  d['client_learning_rate']),
                                              CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER,
                                              lambda d: f"_clr_{d['client_learning_rate']}_slr_"
                                                        f"{d['server_learning_rate']}")
        pgr = pgmg.generate_grid_responses()

        SimulateExperiment.start_experiment(
            f"Cifar10_Lda_{concentration}_Fedadam_HS",
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
            rounds_between_centralised_evaluations=10,
            simulator_provider=SerialExecutionSimulator,
            simulator_args={})
