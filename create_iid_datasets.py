import argparse
import sys
from pathlib import Path

from sources.global_data_properties import DATASET_NAME_LIST
from sources.dataset_utils.create_iid_dataset_utils import create_iid_dataset, \
    DATASET_NAME_FRACTION_DICT, get_default_iid_dataset_filename, \
    get_fractional_iid_dataset_filename, \
    get_full_iid_dataset_filename

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).resolve().parents[0]))

    dataset_identifier_parser = argparse.ArgumentParser(
        description='Select which datasets you wish to create iid_data for. '
                    'Ensure that you have created datasets via initialise_datasets before.')

    dataset_identifiers = DATASET_NAME_LIST

    dataset_identifier_parser.add_argument(
        f'--all',
        action='store_true',
        help=f'Create all default/full datasets.')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}',
            action='store_true',
            help=f'Creates a non-iid {dataset_identifier} dataset using the default fraction.')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}_frac',
            action='store',
            help=f'Creates a non-iid {dataset_identifier} dataset using a specific fraction.')

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}_full',
            action='store_true',
            help=f'Creates a non-iid {dataset_identifier} using the entire dataset.')

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
        print("Data has not yet been downloaded. Please execute the initialise_datasets script "
              "first.")
        sys.exit(0)

    datasets = DATASET_NAME_LIST

    for dataset_identifier in datasets:
        if args[dataset_identifier] or args["all"]:
            print(f"Creating default IID Dataset for {dataset_identifier}")
            create_iid_dataset(
                data_dir,
                dataset_identifier,
                DATASET_NAME_FRACTION_DICT[dataset_identifier],
                get_default_iid_dataset_filename(dataset_identifier)
            )
            print(f"Done: Finished creating default IID Dataset for {dataset_identifier}")

    for dataset_identifier in datasets:
        var_id = dataset_identifier + "_frac"
        if args[var_id] is not None:
            frac = float(args[var_id])
            print(f"Creating fractional ({frac}) IID Dataset for {dataset_identifier}")
            create_iid_dataset(
                data_dir,
                dataset_identifier,
                frac,
                get_fractional_iid_dataset_filename(dataset_identifier, frac)
            )
            print(f"Done: Finished creating fractional ({frac}) IID Dataset for"
                  f" {dataset_identifier}!")

    for dataset_identifier in datasets:
        var_id = dataset_identifier + "_full"
        if args[var_id] or args["all"]:
            print(f"Creating Full IID Dataset for {dataset_identifier}")
            create_iid_dataset(
                data_dir,
                dataset_identifier,
                1.0,
                get_full_iid_dataset_filename(dataset_identifier)
            )
            print(f"Done: Finished creating Full IID Dataset for {dataset_identifier}")