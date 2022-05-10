import functools
import math
from pathlib import Path

from experiments.celeba_experiments.celeba_metadata_providers import \
    CELEBA_BASE_METADATA_REM_EXP_PROVIDER
from sources.datasets.celeba.celeba_client_dataset_factory import CelebaClientDatasetFactory
from sources.datasets.celeba.celeba_client_dataset_processor import CelebaClientDatasetProcessor
from sources.models.celeba.celeba_model_template import CelebaKerasModelTemplate

from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator

from sources.models.model_template_decorators.add_proximal_loss_model_template_decorator import \
    AddProximalLossModelTemplateDecorator

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED


def e(exp):
    return math.pow(10, exp)


def celeba_fedprox():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"
    mu_range = [e(-3), e(-2), e(-1), e(0)]

    for mu in mu_range:
        model_template = AddProximalLossModelTemplateDecorator(CelebaKerasModelTemplate(DEFAULT_SEED),
                                                               mu=mu)
        dataset_factory = CelebaClientDatasetFactory(root_data_dir)
        central_dataset = get_default_iid_dataset("celeba")

        eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
            model_template,
            central_dataset,
            CelebaClientDatasetProcessor())

        initial_parameters = model_template.get_model().get_weights()

        # Args: https://github.com/yjlee22/FedShare/blob/9a8e89b6975cd505005fc79a0b0add72351bab9c/utils/options.py
        strategy_provider_fed_avg = functools.partial(
            full_eval_fed_avg_strategy_provider,
            eval_fn,
            initial_parameters=initial_parameters
        )

        SimulateExperiment.start_experiment(
            f"Celeba_FedProx_Varying_Mu_{mu}",
            model_template,
            dataset_factory,
            strategy_provider=None,
            strategy_provider_list=[strategy_provider_fed_avg],
            optimizer_list=[model_template.get_optimizer(e(-2))],
            experiment_metadata_list=[CELEBA_BASE_METADATA_REM_EXP_PROVIDER(
                custom_suffix=f"_mu_{mu}"
            )],
            base_dir=base_dir,
            runs_per_experiment=1,
            centralised_evaluation=True,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=10,
            simulator_provider=SerialExecutionSimulator
        )
