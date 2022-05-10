import copy
import functools
import math
from pathlib import Path
from typing import List
from numpy import ndarray

from sources.datasets.client_dataset_definitions.client_dataset_decorators.extend_dataset_ratio_based_shared_memory import \
    ExtendDatasetRatioBasedSharedMemory
from sources.metrics.default_metrics_tf import DEFAULT_METRICS
from experiments.cifar10_experiments.cifar10_metadata_providers import \
    CIFAR10_BASE_METADATA_REM_EXP_PROVIDER
from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from sources.datasets.client_dataset_factory_definitions.client_dataset_factory_decorator import \
    DecoratedClientDatasetFactoryDecorator

from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10, \
    DEFAULT_RATIOS_DATASET_SIZE_GD_PARTITION
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import \
    Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import \
    Cifar10LdaClientDatasetProcessor
from sources.models.cifar10_lda.cifar10_lda_model_template import Cifar10LdaKerasModelTemplate

from sources.dataset_creation_utils.get_iid_dataset_utils import get_default_iid_dataset, \
    get_globally_shared_iid_dataset, \
    preprocess_dataset
from sources.metrics.central_evaluation_keras import \
    create_central_evaluation_function_from_dataset_processor_keras
from sources.flwr.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.utils.set_random_seeds import DEFAULT_SEED

NUM_PRETRAIN_EPOCHS = 25


def e(exp):
    return math.pow(10, exp)


def pretrain_model_weights(model_template, globally_shared_dataset, optimizer) -> List[ndarray]:
    model = model_template.get_model()
    model.compile(optimizer,
                  model_template.get_loss(),
                  DEFAULT_METRICS)
    model.fit(x=globally_shared_dataset.train["x"], y=globally_shared_dataset.train["y"],
              epochs=NUM_PRETRAIN_EPOCHS)
    return model.get_weights()


def gsd_cifar10():
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    model_template = Cifar10LdaKerasModelTemplate(DEFAULT_SEED)
    initial_parameters = model_template.get_model().get_weights()

    for concentration in [100.0]:
        # Get Globally Shared Dataset and load it into Ray
        globally_shared_dataset = get_globally_shared_iid_dataset(
            get_lda_cifar10_dataset_name(concentration, 100))

        for ratio in [0.1]:

            # Create Model Template and Smart Data Factory
            model_template = Cifar10LdaKerasModelTemplate(DEFAULT_SEED)
            dataset_factory = Cifar10LdaClientDatasetFactory(root_data_dir, 100, concentration)
            dataset_factory = DecoratedClientDatasetFactoryDecorator(
                dataset_factory,
                ExtendDatasetRatioBasedSharedMemory,
                {"client_gsd_partition_sz_ratios": (ratio, 0, 0),
                 "shared_dataset": preprocess_dataset(globally_shared_dataset,
                                                      Cifar10LdaClientDatasetProcessor()
                                                      )})

            central_dataset = get_default_iid_dataset(get_lda_cifar10_dataset_name(concentration, 100))

            eval_fn = create_central_evaluation_function_from_dataset_processor_keras(
                model_template,
                central_dataset,
                Cifar10LdaClientDatasetProcessor())

            optimizer_fed_avg = model_template.get_optimizer(1e-1)

            # Pretrain Model on Dataset for 25 Epochs


            optimizer_fed_avg = model_template.get_optimizer(1e-1)

            # Args: https://github.com/yjlee22/FedShare/blob/9a8e89b6975cd505005fc79a0b0add72351bab9c/utils/options.py
            strategy_provider_fed_avg = functools.partial(
                full_eval_fed_avg_strategy_provider,
                eval_fn,
                initial_parameters=copy.deepcopy(initial_parameters)
            )

            SimulateExperiment.start_experiment(
                f"Cifar10Lda_{concentration}_SharedGlobalDataset_Ratio_{str(ratio)}_No_Pretraining",
                model_template,
                dataset_factory,
                strategy_provider=None,
                strategy_provider_list=[strategy_provider_fed_avg],
                optimizer_list=[optimizer_fed_avg],
                experiment_metadata_list=[CIFAR10_BASE_METADATA_REM_EXP_PROVIDER(
                    custom_suffix=f"_r_{str(ratio)}")],
                base_dir=base_dir,
                runs_per_experiment=1,
                centralised_evaluation=True,
                aggregated_evaluation=True,
                rounds_between_centralised_evaluations=10,
            )
