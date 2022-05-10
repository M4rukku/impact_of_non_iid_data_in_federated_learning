import functools
import math
from pathlib import Path
from sources.models.model_template_decorators.add_proximal_loss_model_template_decorator import \
    AddProximalLossModelTemplateDecorator
from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from experiments.cifar10_experiments.cifar10_metadata_providers import CIFAR10_BASE_METADATA_REM_EXP_PROVIDER
import tensorflow as tf
from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import Cifar10LdaClientDatasetProcessor
from sources.models.cifar10_lda.cifar10_lda_model_template import Cifar10LdaKerasModelTemplate

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED


def e(exp):
    return math.pow(10, exp)

mu_range = [e(-3), e(-2), e(-1), e(0)]

def fed_prox_cifar10():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    for concentration in DEFAULT_CONCENTRATIONS_CIFAR10:
        for mu in mu_range:
            model_template = AddProximalLossModelTemplateDecorator(Cifar10LdaKerasModelTemplate(
                DEFAULT_SEED,
            ), mu=mu)
            dataset_factory = Cifar10LdaClientDatasetFactory(root_data_dir, 100, concentration)
            central_dataset = get_default_iid_dataset(get_lda_cifar10_dataset_name(concentration, 100))

            eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
                model_template,
                central_dataset,
                Cifar10LdaClientDatasetProcessor())

            initial_parameters = model_template.get_model().get_weights()

            # Args: https://github.com/yjlee22/FedShare/blob/9a8e89b6975cd505005fc79a0b0add72351bab9c/utils/options.py
            strategy_provider_fed_avg = functools.partial(
                full_eval_fed_avg_strategy_provider,
                eval_fn,
                initial_parameters=initial_parameters
            )
            optimizer_fed_avg = tf.keras.optimizers.SGD(learning_rate=1e-1)

            SimulateExperiment.start_experiment(
                f"Cifar10Lda_{concentration}_FedProx_mu_{str(mu)}",
                model_template,
                dataset_factory,
                strategy_provider=None,
                strategy_provider_list=[strategy_provider_fed_avg],
                optimizer_list=[optimizer_fed_avg],
                experiment_metadata_list=[CIFAR10_BASE_METADATA_REM_EXP_PROVIDER()],
                base_dir=base_dir,
                runs_per_experiment=1,
                centralised_evaluation=True,
                aggregated_evaluation=True,
                rounds_between_centralised_evaluations=10)
