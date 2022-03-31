import argparse
import sys
from pathlib import Path

from sources.dataset_creation_utils.get_iid_dataset_utils import get_full_iid_dataset
from sources.global_data_properties import DATASET_NAME_DEFAULT_FRACTION_DICT, \
    DATASET_NAME_GLOBAL_SHARED_FRACTION_DICT, CLIENT_DATASETS_NAME_LIST, CLIENT_SUBSET_TO_CONSIDER
from sources.dataset_creation_utils.create_iid_dataset_utils import create_iid_dataset_from_client_fraction, \
    get_default_iid_dataset_filename, \
    get_fractional_iid_dataset_filename, \
    get_full_iid_dataset_filename, subsample_full_iid_datasets, \
    get_globally_shared_iid_dataset_filename

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).resolve().parents[0]))

    dataset_identifier_parser = argparse.ArgumentParser(
        description='Select which datasets you wish to create iid_data for. '
                    'Ensure that you have created datasets via initialise_datasets before.')

    dataset_identifiers = CLIENT_DATASETS_NAME_LIST

    dataset_identifier_parser.add_argument(
        f'--all',
        action='store_true',
        help=f'Create all default/full datasets. (Recommended)')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}',
            action='store_true',
            help=f'Creates a non-iid {dataset_identifier} dataset using the default fraction.')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}_global_shared',
            action='store_true',
            help=f'Creates a non-iid {dataset_identifier} dataset using the default fraction '
                 f'for the globally shared datasets.')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}_frac',
            action='store',
            help=f'Creates a non-iid {dataset_identifier} dataset using a custom fraction.')

    # Execute parse_args()
    args = vars(dataset_identifier_parser.parse_args())

    if any(args.values()):
        print("Creating the following non-iid datasets:",
              ", ".join(
                  [dataset_id for dataset_id, val in args.items()
                   if val is True or isinstance(val, str)]))
        print("This might take a while...")
        print("")
    else:
        print(
            "No Dataset to extract non iid data from selected. Please ask for help (-h) to see "
            "which datasets may be extracted and "
            "add them as optional arguments when executing the script.")
        print("Terminating.")
        sys.exit(0)

    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"

    if not data_dir.exists() or not (data_dir / "celeba").exists() \
            or not (data_dir / "femnist").exists() or not (data_dir / "shakespeare").exists():
        print(
            "Data for LEAF has not yet been downloaded. Please execute the initialise_datasets script "
            "first. Skipping these Datasets.")

    datasets = dataset_identifiers

    for dataset_identifier in datasets:
        var_id = dataset_identifier + "_full"
        iid_filename = get_full_iid_dataset_filename(dataset_identifier)
        if not (data_dir / iid_filename).exists() and (data_dir / dataset_identifier).exists():
            print(f"Creating Full IID Dataset for {dataset_identifier}")
            create_iid_dataset_from_client_fraction(
                data_dir,
                dataset_identifier,
                1.0,
                iid_filename,
                max_client_identifier=CLIENT_SUBSET_TO_CONSIDER[dataset_identifier]
                if dataset_identifier in CLIENT_SUBSET_TO_CONSIDER
                else CLIENT_SUBSET_TO_CONSIDER["default"]
            )
            print(f"Done: Finished creating Full IID Dataset for {dataset_identifier}")

    for dataset_identifier in datasets:
        filename = get_default_iid_dataset_filename(dataset_identifier)
        if (args[dataset_identifier] or args["all"]) \
                and (data_dir / dataset_identifier).exists() \
                and not (data_dir / filename).exists():
            print(f"Creating default IID Dataset for {dataset_identifier}")
            subsample_full_iid_datasets(
                data_dir,
                get_full_iid_dataset(dataset_identifier),
                DATASET_NAME_DEFAULT_FRACTION_DICT[
                    dataset_identifier] if dataset_identifier in DATASET_NAME_DEFAULT_FRACTION_DICT else
                DATASET_NAME_DEFAULT_FRACTION_DICT["default"],
                filename
            )
            print(f"Done: Finished creating default IID Dataset for {dataset_identifier}")

    for dataset_identifier in datasets:
        var_id = dataset_identifier + "_frac"
        if args[var_id] is not None:
            frac = float(args[var_id])
            print(f"Creating fractional ({frac}) IID Dataset for {dataset_identifier}")
            subsample_full_iid_datasets(
                data_dir,
                get_full_iid_dataset(dataset_identifier),
                frac,
                get_fractional_iid_dataset_filename(dataset_identifier, frac)
            )
            print(f"Done: Finished creating fractional ({frac}) IID Dataset for"
                  f" {dataset_identifier}!")

    for dataset_identifier in datasets:
        var_id = dataset_identifier + "_global_shared"
        filename = get_globally_shared_iid_dataset_filename(dataset_identifier)
        if (args[var_id] or args["all"]) is not None \
                and (data_dir / dataset_identifier).exists() \
                and not (data_dir / filename).exists():
            print(f"Creating globally shared IID Dataset for {dataset_identifier}")
            subsample_full_iid_datasets(
                data_dir,
                get_full_iid_dataset(dataset_identifier),
                DATASET_NAME_GLOBAL_SHARED_FRACTION_DICT[dataset_identifier]
                if dataset_identifier in DATASET_NAME_GLOBAL_SHARED_FRACTION_DICT else
                DATASET_NAME_GLOBAL_SHARED_FRACTION_DICT["default"],
                filename,
                True
            )
            print(f"Done: Finished creating globally shared dataset for "
                  f" {dataset_identifier}!")
