import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import PIL
import numpy as np
import pickle

from sources.dataset_utils.create_iid_dataset_utils import get_full_iid_dataset_filename
from sources.dataset_utils.get_iid_dataset_utils import get_full_iid_dataset
from sources.flwr_parameters.set_random_seeds import DEFAULT_SEED
from sources.global_data_properties import DATASET_NAME_LIST


def get_dataset_dist(data_dir, dataset_identifier):
    dataset_dir = data_dir / dataset_identifier
    total = []
    for subdir in dataset_dir.iterdir():
        with (subdir / "train.pickle").open("rb") as f:
            data = pickle.load(f)
        total.append(len(data["x"]))
    total = np.array(total)
    return total


def create_iid_clients(data_dir: Path, dataset_identifier: str, num_clients: Optional[int] = None,
                       empirical=False):

    dist = get_dataset_dist(data_dir, dataset_identifier)
    rng = np.random.default_rng(DEFAULT_SEED)

    dataset_sizes = None

    if empirical:
        dataset_sizes = dist

        if num_clients is not None:
            dataset_sizes = rng.choice(dist, size=num_clients, replace=True)
            dataset_sizes = list(map(int, dataset_sizes))
    else:
        avg = math.ceil(np.average(dist))

        if num_clients is None:
            dataset_sizes = np.full_like(dist, avg, dtype=np.int32)
        else:
            dataset_sizes = np.full(num_clients, avg, dtype=np.int32)

    full_dataset = get_full_iid_dataset(dataset_identifier)

    type_val = np.object_ if isinstance(full_dataset.test["x"][0], PIL.Image.Image) else None

    full_datasets = {
        "train": {
            "x": np.array(full_dataset.train["x"], dtype=type_val),
            "y": np.array(full_dataset.train["y"])
        },
        "test": {
            "x": np.array(full_dataset.test["x"], dtype=type_val),
            "y": np.array(full_dataset.test["y"])
        },
        "val": {
            "x": np.array(full_dataset.validation["x"], dtype=type_val),
            "y": np.array(full_dataset.validation["y"])
        },
    }

    full_dataset_train_length = len(full_dataset.train["x"])
    full_dataset_test_length = len(full_dataset.test["x"])
    full_dataset_val_length = len(full_dataset.validation["x"])

    dirname = data_dir / f"{dataset_identifier}_iid"
    counter = 0

    for dataset_size in dataset_sizes:
        # Assuming Train is 0.6, val 0.2, test 0.2
        num_train = max(dataset_size, 1)
        num_test = max(int(dataset_size / 0.6 * 0.2), 1)
        num_validation = max(int(dataset_size / 0.6 * 0.2), 1)

        train_selection = rng.choice(full_dataset_train_length, size=num_train, replace=False)
        test_selection = rng.choice(full_dataset_test_length, size=num_test, replace=False)
        val_selection = rng.choice(full_dataset_val_length, size=num_validation, replace=False)

        save_dirname = dirname / str(counter)
        save_dirname.mkdir(parents=True, exist_ok=False)

        def create_client_component(split_identifier: str, selection: np.array):
            client_split = {
                "client_id": counter,
                "x": full_datasets[split_identifier]["x"][selection],
                "y": full_datasets[split_identifier]["y"][selection]
            }

            with (save_dirname / f"{split_identifier}.pickle").open("wb") as f:
                pickle.dump(client_split, f)

        create_client_component("train", train_selection)
        create_client_component("test", test_selection)
        create_client_component("val", val_selection)

        counter += 1


if __name__ == '__main__':
    sys.path.append(str(Path(__file__).resolve().parents[0]))
    data_dir = Path(__file__).parent / "data"

    dataset_identifier_parser = argparse.ArgumentParser(
        description='Select which datasets you wish to create iid client data for. '
                    'Ensure that you have created the full iid datasets via create_iid_datasets '
                    '--all before.')

    dataset_identifiers = DATASET_NAME_LIST

    for dataset_identifier in dataset_identifiers:
        if not (data_dir / get_full_iid_dataset_filename(dataset_identifier)).exists():
            print("Error, you have not created all full iid datasets. Please proceed to do so "
                  "first.")
            sys.exit()

    dataset_identifier_parser.add_argument(
        f'--all',
        action='store_true',
        help=f'Create iid clients for all datasets.')

    dataset_identifier_parser.add_argument(
        f'--empirical',
        action='store_true',
        help=f'Use empirical distribution to create datasets.')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}',
            action='store_true',
            help=f'Creates iid clients for {dataset_identifier}')

    args = vars(dataset_identifier_parser.parse_args())

    for dataset_identifier in dataset_identifiers:
        if args[dataset_identifier] or args["all"]:
            create_iid_clients(data_dir, dataset_identifier, empirical=args["empirical"])
