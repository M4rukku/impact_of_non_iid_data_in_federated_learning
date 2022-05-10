import functools
import math
from pathlib import Path

from experiments.shakespeare_experiments.shakespeare_metadata_providers import \
    SHAKESPEARE_BASE_METADATA_OPT_EXP_PROVIDER
from sources.datasets.shakespeare.shakespeare_client_dataset_factory import \
    ShakespeareClientDatasetFactory
from sources.datasets.shakespeare.shakespeare_client_dataset_processor import \
    ShakespeareClientDatasetProcessor
from sources.models.shakespeare.shakespeare_model_template import ShakespeareKerasModelTemplate

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider, full_eval_fed_adam_strategy_provider, \
    full_eval_fed_adagrad_strategy_provider, \
    full_eval_fed_yogi_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED
from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator


def e(exp):
    return math.pow(10, exp)


def shakespeare_vo():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = ShakespeareKerasModelTemplate(DEFAULT_SEED)
    dataset_factory = ShakespeareClientDatasetFactory(root_data_dir)

    central_dataset = get_default_iid_dataset("shakespeare")

    eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
        model_template,
        central_dataset,
        ShakespeareClientDatasetProcessor())

    initial_model = model_template.get_model()
    initial_parameters = initial_model.get_weights()

    strategy_provider_fed_avg = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn,
        initial_parameters=initial_parameters
    )
    optimizer_fed_avg = model_template.get_optimizer(0.8)

    strategy_provider_fed_adam = functools.partial(
        full_eval_fed_adam_strategy_provider,
        eval_fn,
        initial_parameters=initial_parameters,
        eta=e(-2),
        eta_l=e(-1),
        beta_1=0.9,
        beta_2=0.99,
        tau=e(-3)
    )
    optimizer_fed_adam = model_template.get_optimizer(e(-1))

    strategy_provider_fed_adagrad = functools.partial(
        full_eval_fed_adagrad_strategy_provider,
        eval_fn,
        initial_parameters=initial_parameters,
        eta=e(-1),
        eta_l=e(-1),
        tau=e(-1)
    )
    optimizer_fed_adagrad = model_template.get_optimizer(e(-1))

    strategy_provider_fed_yogi = functools.partial(
        full_eval_fed_yogi_strategy_provider,
        eval_fn,
        initial_parameters=initial_parameters,
        beta_1=0.9,
        beta_2=0.99,
        eta=e(-2),
        eta_l=e(-1),
        tau=e(-3)
    )
    optimizer_fed_yogi = model_template.get_optimizer(e(-1))

    strategy_providers_list = [strategy_provider_fed_avg, strategy_provider_fed_adam,
                               strategy_provider_fed_adagrad, strategy_provider_fed_yogi]
    optimizers_list = [optimizer_fed_avg, optimizer_fed_adam,
                       optimizer_fed_adagrad, optimizer_fed_yogi]

    experiment_metadata_list = \
        [SHAKESPEARE_BASE_METADATA_OPT_EXP_PROVIDER(custom_suffix="_fedavg"),
         SHAKESPEARE_BASE_METADATA_OPT_EXP_PROVIDER(custom_suffix="_fedadam"),
         SHAKESPEARE_BASE_METADATA_OPT_EXP_PROVIDER(custom_suffix="_fedadagrad"),
         SHAKESPEARE_BASE_METADATA_OPT_EXP_PROVIDER(custom_suffix="_fedyogi")]

    SimulateExperiment.start_experiment(
        f"Shakespeare_Varying_Optimisers",
        model_template,
        dataset_factory,
        strategy_provider=None,
        strategy_provider_list=strategy_providers_list,
        optimizer_list=optimizers_list,
        experiment_metadata_list=experiment_metadata_list,
        base_dir=base_dir,
        runs_per_experiment=2,
        centralised_evaluation=True,
        aggregated_evaluation=True,
        rounds_between_centralised_evaluations=2,
        simulator_provider=SerialExecutionSimulator,
        simulator_args={})
