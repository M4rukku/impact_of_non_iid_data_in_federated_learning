import pickle
from os import PathLike
from pathlib import Path

import numpy as np

EVALUATION_METRICS_BASE_FILENAME = "evaluation_metrics_"
CENTRALISED_EVALUATION_METRICS_BASE_FILENAME = "centralised_evaluation_metrics_"


def create_evaluation_metrics_filename(saving_dir: PathLike,
                                       experiment_identifier: str):
    return Path(saving_dir) \
           / f"{EVALUATION_METRICS_BASE_FILENAME}{experiment_identifier}.csv"


def create_round_based_evaluation_metrics_filename(rnd: int,
                                                   saving_dir: PathLike,
                                                   experiment_identifier: str):
    return Path(saving_dir) / \
           f"{EVALUATION_METRICS_BASE_FILENAME}{experiment_identifier}_{str(rnd)}.pkl"


def create_round_based_centralised_evaluation_metrics_filename(rnd: int,
                                                               saving_dir: PathLike,
                                                               experiment_identifier: str):
    return Path(saving_dir) / \
           f"{CENTRALISED_EVALUATION_METRICS_BASE_FILENAME}{experiment_identifier}_{str(rnd)}.pkl"


MODEL_SAVING_BASE_FILENAME = "model_"


def create_round_based_model_saving_filename(rnd: int,
                                             logging_dir: PathLike,
                                             experiment_identifier: str):
    return Path(logging_dir) / \
           f"{MODEL_SAVING_BASE_FILENAME}{experiment_identifier}_{str(rnd)}.npz"


def ensure_dir_of_file_exists(filepath: Path):
    dir_path = filepath.parent
    dir_path.mkdir(parents=True, exist_ok=True)


def pickle_parameters_to_file(filepath: Path, parameters: object):
    ensure_dir_of_file_exists(filepath)
    with filepath.open("wb") as f:
        pickle.dump(parameters, f)


def npz_parameters_to_file(filepath: Path, parameters: np.array):
    ensure_dir_of_file_exists(filepath)
    np.savez(str(filepath), parameters)


def append_data_to_file(filepath: Path, datapoint: str):
    ensure_dir_of_file_exists(filepath)

    if filepath.exists():
        with filepath.open("a") as f:
            f.write(f",{datapoint}")
    else:
        with filepath.open("a") as f:
            f.write(f"{datapoint}")
