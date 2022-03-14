import argparse
import logging
import sys
import pickle
from pathlib import Path
from operator import itemgetter
import tensorflow_datasets as tfds
import numpy as np
from flwr.dataset.utils.common import create_lda_partitions

from sources.dataset_utils.create_iid_dataset_utils import get_full_iid_dataset_filename, \
    get_default_iid_dataset_filename, get_fractional_iid_dataset_filename
from sources.dataset_utils.create_lda_dataset_utils import get_random_id_splits, \
    get_lda_cifar10_dataset_name
from sources.dataset_utils.dataset import Dataset
from sources.flwr_parameters.set_random_seeds import set_seeds, DEFAULT_SEED
from sources.global_data_properties import DEFAULT_CONCENTRATIONS_CIFAR10, TOTAL_IMAGES_CIFAR10, DEFAULT_PARTITIONS_CIFAR10, \
    DEFAULT_TRAIN_SPLIT, DEFAULT_TEST_SPLIT, DEFAULT_VALIDATION_SPLIT, DEFAULT_IID_DATASET_SIZE_CIFAR10

fst = itemgetter(0)
snd = itemgetter(1)

# Aims to implement https://arxiv.org/pdf/1909.06335.pdf
default_images_per_client = TOTAL_IMAGES_CIFAR10 / DEFAULT_PARTITIONS_CIFAR10

default_accept_imbalanced = True


def create_iid_cifar10_dataset(iid_dataset_filepath: Path,
                               fraction_to_subsample: float,
                               cifar10_x_array,
                               cifar10_y_array):
    print(f"Creating iid dataset file {iid_dataset_filepath.stem}")

    if iid_dataset_filepath.exists():
        print("Dataset already exists... Returning")
        return

    dataset_size = len(cifar10_numpy_x)
    subsample_size = int(dataset_size * fraction_to_subsample)
    rng = np.random.default_rng(DEFAULT_SEED)

    dataset_index_selection = rng.choice(dataset_size,
                                         subsample_size,
                                         replace=False)

    dataset_selection_x = cifar10_x_array[dataset_index_selection]
    dataset_selection_y = cifar10_y_array[dataset_index_selection]

    train_split, test_split = (int(DEFAULT_TRAIN_SPLIT * dataset_size),
                               int((DEFAULT_TRAIN_SPLIT + DEFAULT_TEST_SPLIT) * dataset_size))

    train_test_validation_dataset = Dataset(
        {"x": dataset_selection_x[:train_split], "y": dataset_selection_y[:train_split]},
        {"x": dataset_selection_x[train_split:test_split], "y": dataset_selection_y[train_split:test_split]},
        {"x": dataset_selection_x[test_split:], "y": dataset_selection_y[test_split:]})

    with iid_dataset_filepath.open("wb") as f:
        pickle.dump(train_test_validation_dataset.to_tuple(), f)


if __name__ == '__main__':
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    dataset_identifier_parser = argparse.ArgumentParser(
        description=f'This tools creates client datasets from Cifar10 based on a latent dirichlet '
                    f'distribution. By default we will create 100 clients with 500 samples each. '
                    f'We choose the following concentration parameters for lda sampling: '
                    f'{str(DEFAULT_CONCENTRATIONS_CIFAR10)}. You can change the default behaviour by '
                    f'manually entering the number of clients and/or cancentrations that will get '
                    f'assigned the total (60k) samples in Cifar10. Note that we use a 80/15/5 split'
                    f''
                    f'You will need to write --default to start the script for the creation of '
                    f'the default values. Otherwise specify --num_partitions or --concentrations')

    dataset_identifier_parser.add_argument(
        f'--start',
        action='store_true',
        help=f'Create client datasets with the default parameters. This option must be used to '
             f'ensure that starting the process is valid')

    dataset_identifier_parser.add_argument(
        f'--iid_datasets',
        action='store_true',
        help=f'Create IID Datasets for the different tasks we will simulate')

    dataset_identifier_parser.add_argument(
        f'--custom_iid_fractions',
        action='store',
        type=float,
        default=[],
        nargs='+',
        help=f'Defines custom fractions, which will decide the size of the iid datasets we will subsample'
             f'created. i.e. 0.5 => 0.5 * 60k = 30k samples.')

    dataset_identifier_parser.add_argument(
        f'--num_partitions',
        action='store',
        type=int,
        default=DEFAULT_PARTITIONS_CIFAR10,
        help=f'Define the number of clients to be generated from the cifar10 dataset. If not '
             f'defined, will be set to default value ({str(DEFAULT_PARTITIONS_CIFAR10)}).')

    dataset_identifier_parser.add_argument(
        f'--concentrations',
        action='store',
        type=float,
        nargs='+',
        default=DEFAULT_CONCENTRATIONS_CIFAR10,
        help=f'Define concentrations to be used in lda sampling. Can assign multiple values i.e. '
             f'--conventrations 0.5 0.6 0.3. By default will be set to {str(DEFAULT_CONCENTRATIONS_CIFAR10)}')

    args = vars(dataset_identifier_parser.parse_args())

    num_partitions = args["num_partitions"]
    concentrations = args["concentrations"]

    iid_datasets = args["iid_datasets"]
    custom_iid_fractions = args["custom_iid_fractions"]

    if not args["start"]:
        print("If you wish to proceed with creating lda clients for cifar10, please add the start "
              "flag to the arguments. For more information use --help")
        sys.exit(0)

    # Load Dataset
    cifar10_dataset = tfds.load(
        "cifar10",
        split="train[0:100%]+test[0:100%]",
        shuffle_files=True,
        as_supervised=True
    )
    cifar10_dataset_list = list(cifar10_dataset.as_numpy_iterator())
    cifar10_numpy_x = np.array(list(map(fst, cifar10_dataset_list)))
    cifar10_numpy_y = np.array(list(map(snd, cifar10_dataset_list)))

    dataset_size = len(cifar10_numpy_x)

    if iid_datasets:
        # Create Full IID Dataset 80/15/5 Split
        full_iid_dataset_name = get_full_iid_dataset_filename("cifar10")
        full_iid_dataset_file = data_dir / full_iid_dataset_name
        create_iid_cifar10_dataset(full_iid_dataset_file, 1.0, cifar10_numpy_x, cifar10_numpy_y)

        # Create Default Split
        default_iid_dataset_identifier = get_default_iid_dataset_filename("cifar10")
        default_iid_dataset_file = data_dir / default_iid_dataset_identifier
        create_iid_cifar10_dataset(default_iid_dataset_file, DEFAULT_IID_DATASET_SIZE_CIFAR10, cifar10_numpy_x, cifar10_numpy_y)

    # Create Fractional Splits
    for fraction in custom_iid_fractions:
        default_iid_dataset_identifier = get_fractional_iid_dataset_filename("cifar10", fraction)
        default_iid_dataset_file = data_dir / default_iid_dataset_identifier
        create_iid_cifar10_dataset(default_iid_dataset_file, fraction, cifar10_numpy_x, cifar10_numpy_y)

    # Create IID Clients
    for concentration in concentrations:
        set_seeds()
        base_dirname = get_lda_cifar10_dataset_name(concentration, num_partitions)
        base_dir = data_dir / base_dirname

        if base_dir.exists():
            logging.warning(f"Clients for concentration {concentration} have already been created. "
                            f"If you wish to recreate the datasets, please delete the old folder first.")
            continue

        lda_partitions = create_lda_partitions((cifar10_numpy_x, cifar10_numpy_y),
                                               num_partitions=num_partitions,
                                               concentration=concentration,
                                               accept_imbalanced=default_accept_imbalanced)

        xy_datasets = lda_partitions[0]

        base_dir.mkdir(parents=True, exist_ok=False)
        with (base_dir / "lda_client_distribution").open("wb") as f:
            pickle.dump(lda_partitions[1], f)

        for i, (x_data, y_data) in enumerate(xy_datasets):
            data_length = len(y_data)
            train_idx, test_idx, val_idx = get_random_id_splits(
                data_length,
                DEFAULT_TEST_SPLIT,
                DEFAULT_VALIDATION_SPLIT)

            dataset_splits = {
                "train": {"x": x_data[train_idx], "y": y_data[train_idx], "client_id": i},
                "test": {"x": x_data[test_idx], "y": y_data[test_idx], "client_id": i},
                "val": {"x": x_data[val_idx], "y": y_data[val_idx], "client_id": i}
            }

            save_dirname = base_dir / str(i)
            save_dirname.mkdir(parents=True, exist_ok=False)
            for split_identifier, split_data_dir in dataset_splits.items():
                with (save_dirname / f"{split_identifier}.pickle").open("wb") as f:
                    pickle.dump(split_data_dir, f)
