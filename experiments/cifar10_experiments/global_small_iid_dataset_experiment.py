import functools
import math
from pathlib import Path
from typing import List

import experiments.setup_system_paths as ssp

ssp.setup_system_paths()

from numpy import ndarray
from sources.metrics.default_metrics import DEFAULT_METRICS
from experiments.cifar10_experiments.cifar10_metadata_providers import CIFAR10_BASE_METADATA_REM_EXP_PROVIDER
from sources.dataset_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from sources.datasets.client_dataset_definitions.client_dataset_decorators.extend_dataset_ray_decorator import \
    ExtendDatasetRayDecorator
from sources.datasets.client_dataset_factory_definitions.client_dataset_factory_decorator import \
    DecoratedClientDatasetFactoryDecorator
from sources.ray_tooling.dataset_management_for_ray import load_dataset_into_ray

from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10, \
    NUM_DATA_SAMPLES_USED_PER_CLIENT_FROM_GLOBAL_DATASET_CIFAR10
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import Cifar10LdaClientDatasetProcessor
from sources.models.cifar10_lda.cifar10_lda_model_template import Cifar10LdaModelTemplate

from sources.dataset_utils.get_iid_dataset_utils import get_default_iid_dataset, get_globally_shared_iid_dataset, \
    preprocess_dataset
from sources.metrics.central_evaluation import \
    create_central_evaluation_function_from_dataset_processor
from sources.flwr_strategies.full_evaluation_strategy_providers import \
    full_eval_fed_avg_strategy_provider

from sources.experiments.simulate_experiment import SimulateExperiment
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED

NUM_PRETRAIN_EPOCHS = 30

def e(exp):
    return math.pow(10, exp)


def pretrain_model_weights(model_template, globally_shared_dataset, optimizer) -> List[ndarray]:
    model = model_template.get_model()
    model.compile(optimizer,
                  model_template.get_loss(),
                  DEFAULT_METRICS)
    model.fit(x=globally_shared_dataset.train["x"], y=globally_shared_dataset.train["y"], epochs=NUM_PRETRAIN_EPOCHS)
    return model.get_weights()


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    for concentration in DEFAULT_CONCENTRATIONS_CIFAR10:
        # Get Globally Shared Dataset and load it into Ray
        globally_shared_dataset = get_globally_shared_iid_dataset(get_lda_cifar10_dataset_name(concentration, 100))
        ray_callbacks = [functools.partial(load_dataset_into_ray,
                                           "cifar10_global_shared",
                                           globally_shared_dataset,
                                           Cifar10LdaClientDatasetProcessor())]

        # Create Model Template and Smart Data Factory
        model_template = Cifar10LdaModelTemplate(DEFAULT_SEED)
        dataset_factory = Cifar10LdaClientDatasetFactory(root_data_dir, 100, concentration)
        dataset_factory = DecoratedClientDatasetFactoryDecorator(
            dataset_factory,
            ExtendDatasetRayDecorator,
            {"dataset_identifier": "cifar10_global_shared",
             "subset_size": (NUM_DATA_SAMPLES_USED_PER_CLIENT_FROM_GLOBAL_DATASET_CIFAR10, 0, 0),
             "shared_dataset_size": (
                 len(globally_shared_dataset.train["x"]),
                 len(globally_shared_dataset.test["x"]),
                 len(globally_shared_dataset.validation["x"])
             )})

        total_clients = dataset_factory.get_number_of_clients()
        central_dataset = get_default_iid_dataset(get_lda_cifar10_dataset_name(concentration, 100))

        eval_fn = create_central_evaluation_function_from_dataset_processor(
            model_template,
            central_dataset,
            Cifar10LdaClientDatasetProcessor())

        optimizer_fed_avg = model_template.get_optimizer(1e-1, 0.5)

        # Pretrain Model on Dataset for 25 Epochs
        initial_parameters = pretrain_model_weights(model_template, preprocess_dataset(globally_shared_dataset,
                                                                                       Cifar10LdaClientDatasetProcessor()),
                                                    optimizer_fed_avg)

        optimizer_fed_avg = model_template.get_optimizer(1e-1, 0.5)

        # Args: https://github.com/yjlee22/FedShare/blob/9a8e89b6975cd505005fc79a0b0add72351bab9c/utils/options.py
        strategy_provider_fed_avg = functools.partial(
            full_eval_fed_avg_strategy_provider,
            eval_fn,
            initial_parameters=initial_parameters
        )

        SimulateExperiment.start_experiment(
            f"Cifar10Lda_{concentration}_SharedGlobalDataset",
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
            rounds_between_centralised_evaluations=10,
            ray_callbacks=ray_callbacks)
