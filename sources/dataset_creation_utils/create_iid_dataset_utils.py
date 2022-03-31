import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from sources.utils.dataset import Dataset
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED


def extract_data(dataset_path: Path):
    train_set = dataset_path / "train.pickle"
    test_set = dataset_path / "test.pickle"
    validation_set = dataset_path / "val.pickle"

    with train_set.open("rb") as datafile:
        train_data = pickle.load(datafile)
    with test_set.open("rb") as datafile:
        test_data = pickle.load(datafile)
    with validation_set.open("rb") as datafile:
        validation_data = pickle.load(datafile)

    return train_data, test_data, validation_data


def append_data(train_data, test_data, validation_data,
                new_train_data, new_test_data, new_validation_data):
    if isinstance(train_data["x"], list) and isinstance(new_train_data["x"], list):
        train_data["x"] = train_data["x"] + new_train_data["x"]
        train_data["y"] = train_data["y"] + new_train_data["y"]
        test_data["x"] = test_data["x"] + new_test_data["x"]
        test_data["y"] = test_data["y"] + new_test_data["y"]
        validation_data["x"] = validation_data["x"] + new_validation_data["x"]
        validation_data["y"] = validation_data["y"] + new_validation_data["y"]
    else:
        train_data["x"] = np.concatenate([train_data["x"], new_train_data["x"]])
        train_data["y"] = np.concatenate([train_data["y"], new_train_data["y"]])
        test_data["x"] = np.concatenate([test_data["x"], new_test_data["x"]])
        test_data["y"] = np.concatenate([test_data["y"], new_test_data["y"]])
        validation_data["x"] = np.concatenate([validation_data["x"], new_validation_data["x"]])
        validation_data["y"] = np.concatenate([validation_data["y"], new_validation_data["y"]])


def dataset_to_numpy_arrays(dataset: Dataset) -> Dataset:
    return Dataset(
        {"x": np.array(dataset.train["x"]), "y": np.array(dataset.train["y"])},
        {"x": np.array(dataset.test["x"]), "y": np.array(dataset.test["y"])},
        {"x": np.array(dataset.validation["x"]), "y": np.array(dataset.validation["y"])},
    )


def create_iid_dataset_from_client_fraction(base_data_dir: Path,
                                            dataset_identifier: str,
                                            fraction_to_extract: float,
                                            compiled_dataset_name: str = None,
                                            only_create_and_use_training_data: str = False,
                                            max_client_identifier: Optional[int] = None
                                            ) -> None:
    dataset_dir = base_data_dir / dataset_identifier
    dataset_files = np.array(list(filter(lambda file: file.is_dir(), dataset_dir.iterdir())))
    num_clients_to_consider: int = len(dataset_files) if max_client_identifier is None \
        else min(max_client_identifier, len(dataset_files))
    amt_files_to_extract = int(fraction_to_extract * num_clients_to_consider)

    rng = np.random.default_rng(DEFAULT_SEED)
    dataset_index_selection = rng.choice(num_clients_to_consider,
                                         amt_files_to_extract,
                                         replace=False)

    dataset_selection = dataset_files[dataset_index_selection]

    dataset = extract_data(dataset_selection[0])

    def empty_x():
        arr = dataset[0]["x"]
        if isinstance(arr, list):
            return []
        shape_x = arr.shape
        shape_x = (0, *(shape_x[1:]))
        return np.empty(shape_x, arr.dtype)

    def empty_y():
        arr = dataset[0]["y"]
        if isinstance(arr, list):
            return []
        shape_y = arr.shape
        shape_y = (0, *(shape_y[1:]))
        return np.empty(shape_y, arr.dtype)

    train_test_validation_dataset = Dataset({"x": empty_x(), "y": empty_y()},
                                            {"x": empty_x(), "y": empty_y()},
                                            {"x": empty_x(), "y": empty_y()})

    for dataset in dataset_selection:
        extracted_data = extract_data(dataset)
        if only_create_and_use_training_data:
            append_data(*train_test_validation_dataset, extracted_data[0],
                        {"x": empty_x(), "y": empty_y()},
                        {"x": empty_x(), "y": empty_y()})
        else:
            append_data(*train_test_validation_dataset, *extracted_data)

    train_test_validation_dataset = dataset_to_numpy_arrays(train_test_validation_dataset)

    if compiled_dataset_name is None:
        iid_data_file = base_data_dir / \
                        f"{dataset_identifier}_iid_dataset_fraction_{fraction_to_extract}.pkl"
    else:
        iid_data_file = base_data_dir / compiled_dataset_name

    with iid_data_file.open("wb") as f:
        pickle.dump(train_test_validation_dataset.to_tuple(), f)


def subsample_full_iid_datasets(base_data_dir: Path,
                                full_dataset: Dataset,
                                fraction_to_extract: float,
                                compiled_dataset_name: str,
                                only_create_and_use_training_data: bool = False):
    rng = np.random.default_rng(DEFAULT_SEED)
    selection_train = rng.choice(len(full_dataset.train["x"]),
                                 int(len(full_dataset.train["x"]) * fraction_to_extract),
                                 replace=False)
    selection_test = rng.choice(len(full_dataset.test["x"]),
                                int(len(full_dataset.test["x"]) * fraction_to_extract),
                                replace=False)
    selection_validation = rng.choice(len(full_dataset.validation["x"]),
                                      int(len(full_dataset.validation["x"]) * fraction_to_extract),
                                      replace=False)

    if only_create_and_use_training_data:
        def empty_x():
            arr = full_dataset.train["x"]
            if isinstance(arr, list):
                return []
            shape_x = arr.shape
            shape_x = (0, *(shape_x[1:]))
            return np.empty(shape_x, arr.dtype)

        def empty_y():
            arr = full_dataset.train["y"]
            if isinstance(arr, list):
                return []
            shape_y = arr.shape
            shape_y = (0, *(shape_y[1:]))
            return np.empty(shape_y, arr.dtype)

        train_test_validation_dataset = Dataset({"x": full_dataset.train["x"][selection_train],
                                                 "y": full_dataset.train["y"][selection_train]},
                                                {"x": empty_x(), "y": empty_y()},
                                                {"x": empty_x(), "y": empty_y()})
    else:
        train_test_validation_dataset = Dataset({"x": full_dataset.train["x"][selection_train],
                                                 "y": full_dataset.train["y"][selection_train]},
                                                {"x": full_dataset.test["x"][selection_test],
                                                 "y": full_dataset.test["y"][selection_test]},
                                                {"x": full_dataset.validation["x"][
                                                    selection_validation],
                                                 "y": full_dataset.validation["y"][
                                                     selection_validation]})

    dataset_file = base_data_dir / compiled_dataset_name
    with dataset_file.open("wb") as f:
        pickle.dump(train_test_validation_dataset.to_tuple(), f)


def get_full_iid_dataset_filename(dataset_identifier: str) -> str:
    return dataset_identifier + "_full_iid_dataset.pickle"


def get_globally_shared_iid_dataset_filename(dataset_identifier: str) -> str:
    return dataset_identifier + "_globally_shared.pickle"


def get_default_iid_dataset_filename(dataset_identifier: str) -> str:
    return dataset_identifier + "_default_iid_dataset.pickle"


def get_fractional_iid_dataset_filename(dataset_identifier: str, fraction: float) -> str:
    return f"{dataset_identifier}_frac_{str(fraction)}_iid_dataset.pickle"
