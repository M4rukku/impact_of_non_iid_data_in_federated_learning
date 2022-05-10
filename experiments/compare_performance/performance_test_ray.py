import functools
import math
from pathlib import Path

import os

from sources.simulators.ray_based_simulator import RayBasedSimulator
from sources.simulators.serial_execution_simulator import SerialExecutionSimulator
from sources.utils.simulation_parameters import DEFAULT_RAY_INIT_ARGS

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from experiments.cifar10_experiments.cifar10_metadata_providers import \
    CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER
from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_factory import \
    Cifar10LdaClientDatasetFactory
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import \
    Cifar10LdaClientDatasetProcessor
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


def run_performance_test_ray(run, simulator_name, simulation_provider, simulation_kwargs):
    base_dir = Path(__file__).parent.parent.parent
    root_data_dir = base_dir / "data"

    for concentration in [0.5]:
        model_template = Cifar10LdaKerasModelTemplate(DEFAULT_SEED)
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
            f"Cifar10_Lda_{concentration}_Fedavg_Performance_Test_{simulator_name}_R{run}",
            model_template,
            dataset_factory,
            strategy_provider=None,
            strategy_provider_list=[fed_avg],
            optimizer_list=[model_template.get_optimizer(0.1)],
            experiment_metadata_list=[CIFAR10_BASE_METADATA_SYS_EXP_PROVIDER()],
            base_dir=base_dir,
            runs_per_experiment=1,
            centralised_evaluation=True,
            aggregated_evaluation=True,
            rounds_between_centralised_evaluations=10,
            simulator_provider=simulation_provider,
            simulator_args=simulation_kwargs)


if __name__ == '__main__':
    from time import time, ctime

    simulator_identifiers = ["serial_simulator", "ray_simulator"]
    simulators = {"serial_simulator": SerialExecutionSimulator,
                  "ray_simulator": RayBasedSimulator}
    simulator_kwargs = {"serial_simulator": {},
                        "ray_simulator": {"client_resources": {"num_gpus": 1},
                                          "ray_init_args": {"num_gpus": 1,
                                                            **DEFAULT_RAY_INIT_ARGS}}}
    for simulator in simulator_identifiers:
        runs = 3
        time_delta = 0.0
        deltas = []
        for run in range(runs):
            last_time = time()
            print(f"Evaluation Run {run} starting at {ctime(last_time)}")
            run_performance_test_ray(run,
                                     simulator,
                                     simulators[simulator],
                                     simulator_kwargs[simulator])
            now = time()
            time_delta += now - last_time
            deltas.append(now - last_time)

        average_time_delta = time_delta / runs
        print(f"Results for Simulator: {simulator}")
        print(f"On average, a run with {simulator} took {str(average_time_delta)} seconds")
        print(f"Exact timings are given by {' '.join(map(str, deltas))} seconds")

        min_time = min(deltas)
        max_time = max(deltas)
        print(
            f"Difference between min and av - {average_time_delta - min_time}, difference between "
            f"max and avg - {max_time - average_time_delta}")
        print("")
