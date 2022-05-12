import functools
import math
from pathlib import Path

from experiments.shakespeare_experiments.shakespeare_metadata_providers import \
    SHAKESPEARE_BASE_METADATA_REM_EXP_PROVIDER
from sources.datasets.shakespeare.shakespeare_client_dataset_factory import \
    ShakespeareClientDatasetFactory
from sources.datasets.shakespeare.shakespeare_client_dataset_processor import \
    ShakespeareClientDatasetProcessor
from sources.models.shakespeare.shakespeare_model_template import ShakespeareKerasModelTemplate

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


def shakespeare_fedprox():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"
    mu_range = [e(-3), e(-2), e(-1), e(0)]

    for mu in mu_range:
        model_template = AddProximalLossModelTemplateDecorator(
            ShakespeareKerasModelTemplate(DEFAULT_SEED), mu=mu)
        dataset_factory = ShakespeareClientDatasetFactory(root_data_dir)

        central_dataset = get_default_iid_dataset("shakespeare")

        eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
            ShakespeareKerasModelTemplate(DEFAULT_SEED),
            central_dataset,
            ShakespeareClientDatasetProcessor())

        initial_model = model_template.get_model()
        initial_parameters = initial_model.get_weights()

        # Args: https://github.com/yjlee22/FedShare/blob/9a8e89b6975cd505005fc79a0b0add72351bab9c/utils/options.py
        strategy_provider_fed_avg = functools.partial(
            full_eval_fed_avg_strategy_provider,
            eval_fn,
            initial_parameters=initial_parameters
        )

        SimulateExperiment.start_experiment(
            f"Shakespeare_FedProx_Varying_Mu_{mu}",
            model_template,
            dataset_factory,
            strategy_provider_list=[strategy_provider_fed_avg],
            optimizer_list=[model_template.get_optimizer(0.8)],
            experiment_metadata_list=[SHAKESPEARE_BASE_METADATA_REM_EXP_PROVIDER(
                custom_suffix=f"_mu_{mu}"
            )],
            base_dir=base_dir,
            runs_per_experiment=2,
            centralised_evaluation=True,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=2,
            simulator_provider=SerialExecutionSimulator
        )
