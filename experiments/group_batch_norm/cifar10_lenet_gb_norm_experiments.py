import functools
import math
from pathlib import Path

from experiments.group_batch_norm.gb_norm_metadata_providers import \
    LENET_GB_NORM_EXPERIMENTS_BASE_METADATA_C10
from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import \
    Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import \
    Cifar10LdaClientDatasetProcessor
from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.models.cifar10_lda.cifar10_lenet_bngn_keras_model_template import \
    Cifar10LdaLeNetKerasModelTemplate
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator


def e(exp):
    return math.pow(10, exp)


def run_cifar10_lenet_gb_norm_experiment():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    for GN in [True, False]:
        for concentration in DEFAULT_CONCENTRATIONS_CIFAR10:
            model_template = Cifar10LdaLeNetKerasModelTemplate(DEFAULT_SEED, group_norm=GN)
            dataset_factory = Cifar10LdaClientDatasetFactory(root_data_dir, 100, concentration)
            central_dataset = get_default_iid_dataset(get_lda_cifar10_dataset_name(concentration, 100))

            eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
                model_template,
                central_dataset,
                Cifar10LdaClientDatasetProcessor())

            initial_parameters = model_template.get_model().get_weights()

            fed_avg = functools.partial(
                full_eval_fed_avg_strategy_provider,
                eval_fn,
                initial_parameters=initial_parameters
            )

            SimulateExperiment.start_experiment(
                f"Cifar10_Lda_{concentration}_Fedavg_LeNet_{'GN' if GN else 'BN'}",
                model_template,
                dataset_factory,
                strategy_provider=None,
                strategy_provider_list=[fed_avg],
                optimizer_list=[model_template.get_optimizer(0.002)],
                experiment_metadata_list=[LENET_GB_NORM_EXPERIMENTS_BASE_METADATA_C10(
                    custom_suffix=f"_{'gn' if GN else 'bn'}_lr_0.002")],
                base_dir=base_dir,
                runs_per_experiment=1,
                centralised_evaluation=True,
                aggregated_evaluation=True,
                rounds_between_centralised_evaluations=1,
                simulator_provider=SerialExecutionSimulator,
                simulator_args={})
