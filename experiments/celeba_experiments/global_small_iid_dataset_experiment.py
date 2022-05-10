import copy
import functools
import math
from pathlib import Path
from sources.datasets.celeba.celeba_client_dataset_factory import CelebaClientDatasetFactory

from experiments.celeba_experiments.celeba_metadata_providers import \
    CELEBA_BASE_METADATA_REM_EXP_PROVIDER
from sources.datasets.celeba.celeba_client_dataset_processor import CelebaClientDatasetProcessor
from sources.datasets.client_dataset_definitions.client_dataset_decorators.extend_dataset_ratio_based_shared_memory import \
    ExtendDatasetRatioBasedSharedMemory
from sources.models.celeba.celeba_model_template import CelebaKerasModelTemplate
from sources.simulators.serial_execution_simulator import \
    SerialExecutionSimulator
from sources.metrics.default_metrics_tf import DEFAULT_METRICS

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory_decorator import \
    DecoratedClientDatasetFactoryDecorator

from sources.global_data_properties import \
    DEFAULT_RATIOS_DATASET_SIZE_GD_PARTITION

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset, \
    get_globally_shared_iid_dataset, \
    preprocess_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED

NUM_PRETRAIN_EPOCHS = 30


def e(exp):
    return math.pow(10, exp)


def pretrain_model_weights(model_template, globally_shared_dataset, optimizer):
    model = model_template.get_model()
    model.compile(optimizer,
                  model_template.get_loss(),
                  DEFAULT_METRICS)
    model.fit(x=globally_shared_dataset.train["x"], y=globally_shared_dataset.train["y"],
              epochs=NUM_PRETRAIN_EPOCHS)
    return model


def celeba_fsgd():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"
    model_template = CelebaKerasModelTemplate(DEFAULT_SEED)
    initial_model = model_template.get_model()
        # pretrain_model_weights(model_template,
        #                                    preprocess_dataset(globally_shared_dataset,
        #                                                       CelebaClientDatasetProcessor()),
        #                                    optimizer_fed_avg)
    initial_parameters = initial_model.get_weights()

    for ratio in DEFAULT_RATIOS_DATASET_SIZE_GD_PARTITION:
        # Get Globally Shared Dataset and load it into Ray
        globally_shared_dataset = get_globally_shared_iid_dataset("celeba")

        # Create Model Template and Smart Data Factory
        model_template = CelebaKerasModelTemplate(DEFAULT_SEED)
        dataset_factory = CelebaClientDatasetFactory(root_data_dir)

        dataset_factory = DecoratedClientDatasetFactoryDecorator(
            dataset_factory,
            ExtendDatasetRatioBasedSharedMemory,
            {"client_gsd_partition_sz_ratios": (ratio, 0, 0),
             "shared_dataset": preprocess_dataset(globally_shared_dataset,
                                                  CelebaClientDatasetProcessor()
                                                  )})

        evaluation_dataset = get_default_iid_dataset("celeba")
        eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
            model_template,
            evaluation_dataset,
            CelebaClientDatasetProcessor())

        optimizer_fed_avg = model_template.get_optimizer(e(-2))

        # Pretrain Model on Dataset for 25 Epochs


        # Args: https://github.com/yjlee22/FedShare/blob/9a8e89b6975cd505005fc79a0b0add72351bab9c/utils/options.py
        strategy_provider_fed_avg = functools.partial(
            full_eval_fed_avg_strategy_provider,
            eval_fn,
            initial_parameters=copy.deepcopy(initial_parameters)
        )
        optimizer_fed_avg = model_template.get_optimizer(e(-2))

        SimulateExperiment.start_experiment(
            f"Celeba_SharedGlobalDataset_{str(ratio)}_No_Pretraining",
            model_template,
            dataset_factory,
            strategy_provider=None,
            strategy_provider_list=[strategy_provider_fed_avg],
            optimizer_list=[optimizer_fed_avg],
            experiment_metadata_list=[CELEBA_BASE_METADATA_REM_EXP_PROVIDER(
                custom_suffix=f"_r_{ratio}"
            )],
            base_dir=base_dir,
            runs_per_experiment=1,
            centralised_evaluation=True,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=10,
            simulator_provider=SerialExecutionSimulator)
