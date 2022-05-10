import functools
from pathlib import Path


from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator

from experiments.celeba_experiments.celeba_metadata_providers import \
    CELEBA_BASE_METADATA_SYS_EXP_PROVIDER, VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP
from sources.datasets.celeba.celeba_client_dataset_factory import CelebaClientDatasetFactory
from sources.datasets.celeba.celeba_client_dataset_processor import CelebaClientDatasetProcessor
from sources.models.celeba.celeba_model_template import CelebaKerasModelTemplate
from sources.experiments.grid_search_metadata_provider import ParameterGridMetadataGenerator

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED

def celeba_vle():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = CelebaKerasModelTemplate(DEFAULT_SEED)
    dataset_factory = CelebaClientDatasetFactory(root_data_dir)
    central_dataset = get_default_iid_dataset("celeba")

    eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
        model_template,
        central_dataset,
        CelebaClientDatasetProcessor())

    strategy_provider = functools.partial(
        full_eval_fed_avg_strategy_provider,
        eval_fn
    )

    pgmg = ParameterGridMetadataGenerator(VARYING_LOCAL_EPOCHS_EXP_PARAMETER_MAP,
                                          lambda d: strategy_provider,
                                          lambda d: model_template.get_optimizer(0.01),
                                          CELEBA_BASE_METADATA_SYS_EXP_PROVIDER,
                                          lambda d: f"_le_{d['local_epochs']}")
    pgr = pgmg.generate_grid_responses()

    SimulateExperiment.start_experiment(
        f"Celeba_Varying_Local_Epochs",
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
        simulator_provider=SerialExecutionSimulator)
