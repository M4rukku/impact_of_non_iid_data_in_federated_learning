import argparse
import sys
import pickle
from pathlib import Path
from operator import itemgetter
import tensorflow_datasets as tfds
import numpy as np
from flwr.dataset.utils.common import create_lda_partitions

from sources.dataset_utils.create_lda_dataset_utils import get_random_id_splits, \
    get_lda_dataset_name

fst = itemgetter(0)
snd = itemgetter(1)

# Aims to implement https://arxiv.org/pdf/1909.06335.pdf
default_concentrations = [0.001, 0.5, 100.0]
total_images = 60000
default_num_partitions = 100
default_images_per_client = total_images / default_num_partitions

default_train_split = 0.8
default_test_split = 0.15
default_validation_split = 0.05
default_accept_imbalanced = True

if __name__ == '__main__':
    data_dir = Path(__file__).parent / "data"

    dataset_identifier_parser = argparse.ArgumentParser(
        description=f'This tools creates client datasets from Cifar10 based on a latent dirichlet '
                    f'distribution. By default we will create 100 clients with 500 samples each. '
                    f'We choose the following concentration parameters for lda sampling: '
                    f'{str(default_concentrations)}. You can change the default behaviour by '
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
        f'--num_partitions',
        action='store',
        type=int,
        default=default_num_partitions,
        help=f'Define the number of clients to be generated from the cifar10 dataset. If not '
             f'defined, will be set to default value ({str(default_num_partitions)}).')

    dataset_identifier_parser.add_argument(
        f'--concentrations',
        action='store',
        type=float,
        nargs='+',
        default=default_concentrations,
        help=f'Define concentrations to be used in lda sampling. Can assign multiple values i.e. '
             f'--conventrations 0.5 0.6 0.3. By default will be set to {str(default_concentrations)}')

    args = vars(dataset_identifier_parser.parse_args())

    num_partitions = args["num_partitions"]
    concentrations = args["concentrations"]

    if not args["start"]:
        print("If you wish to proceed with creating lda clients for cifar10, please add the start "
              "flag to the arguments.")
        sys.exit(0)

    cifar10_dataset = tfds.load(
        "cifar10",
        split="train[0:100%]+test[0:100%]",
        shuffle_files=True,
        as_supervised=True
    )
    cifar10_dataset_list = list(cifar10_dataset.as_numpy_iterator())
    cifar10_numpy_x = np.array(list(map(fst, cifar10_dataset_list)))
    cifar10_numpy_y = np.array(list(map(snd, cifar10_dataset_list)))

    for concentration in concentrations:
        lda_partitions = create_lda_partitions((cifar10_numpy_x, cifar10_numpy_y),
                                               num_partitions=num_partitions,
                                               concentration=concentration,
                                               accept_imbalanced=default_accept_imbalanced)

        xy_datasets = lda_partitions[0]

        base_dirname = get_lda_dataset_name(concentration, num_partitions)
        base_dir = data_dir / base_dirname

        base_dir.mkdir(parents=True, exist_ok=False)
        with (base_dir / "lda_client_distribution").open("wb") as f:
            pickle.dump(lda_partitions[1], f)

        for i, (x_data, y_data) in enumerate(xy_datasets):
            data_length = len(y_data)
            train_idx, test_idx, val_idx = get_random_id_splits(
                data_length,
                default_test_split,
                default_validation_split)

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
