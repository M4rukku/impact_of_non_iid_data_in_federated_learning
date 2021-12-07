import argparse
import os
import subprocess
import sys
from pathlib import Path
import zipfile
import gdown


def download_celeba_dataset():
    celeba_download_path = Path(
        __file__).parent / "leaf_root" / "data" / "celeba" / "data" / "raw"
    celeba_download_path.mkdir(parents=True, exist_ok=True)
    celeba_unzip_dir_path = celeba_download_path / "img_align_celeba"

    if celeba_unzip_dir_path.exists():
        print("Celeba Images already exist, skipping download.")
    else:
        print("Downloading Celeba Images from Google")
        celeba_downloaded_zip_path = celeba_download_path / "img_align_celeba.zip"
        gdown.download(
            "https://drive.google.com/uc?id=1SCQmbKY9-zQ7BoQHnkXdW9YrSlnD27xU",
            str(celeba_downloaded_zip_path.resolve()), quiet=False)

        print(f"Successfully downloaded data as {celeba_downloaded_zip_path}")
        print("Unzipping File")

        with zipfile.ZipFile(celeba_downloaded_zip_path, 'r') as zip_ref:
            zip_ref.extractall(celeba_download_path)

        celeba_downloaded_zip_path.unlink()
        print("Successfully Downloaded the Celeba Dataset.")

    # Also get identity / attribute files
    celeba_identity_file_path = celeba_download_path / "identity_CelebA.txt"
    if celeba_identity_file_path.exists():
        print("Celeba Images already exist, skipping download.")
    else:
        print("Downloading Celeba Identity File from Google Drive")
        gdown.download(
            "https://drive.google.com/uc?id=1-kcqScU_Omiqr3STF1SPhuz6lV98knE4",
            str(celeba_identity_file_path.resolve()), quiet=False)

        print(f"Successfully downloaded identity files as "
              f"{celeba_identity_file_path}")

    celeba_list_attr_file_path = celeba_download_path / "list_attr_celeba.txt"
    if celeba_list_attr_file_path.exists():
        print("Celeba Images already exist, skipping download.")
    else:
        print("Downloading Celeba Identity File from Google Drive")
        gdown.download(
            "https://drive.google.com/uc?id=1E1rm_mkvySH6WEwNZgutq5FTs72DKMyx",
            str(celeba_list_attr_file_path.resolve()), quiet=False)

        print(f"Successfully downloaded identity files as "
              f"{celeba_list_attr_file_path}")


if __name__ == '__main__':

    sys.path.append(str(Path(__file__).resolve().parents[0]))

    dataset_identifier_parser = argparse.ArgumentParser(
        description='Select which datasets you wish to download.')

    dataset_identifiers = ["Celeba", "Femnist", "Shakespeare"]

    for dataset_identifier in dataset_identifiers:
        dataset_identifier_parser.add_argument(
            f'--{dataset_identifier.lower()}',
            action='store_true',
            help=f'download the {dataset_identifier} dataset')

    # Execute parse_args()
    args = vars(dataset_identifier_parser.parse_args())

    if any(args.values()):
        print("Downloading the following datasets:",
              ", ".join(
                  [dataset_id for dataset_id, val in args.items() if val]))
        print("This might take a while...")
        print("")
    else:
        print(
            "No Dataset to download selected. Please ask for help (-h) to see "
            "which datasets may be downloaded and "
            "add them as optional arguments when executing the script.")
        print("Terminating.")
        sys.exit(0)

    project_dir = Path(__file__).parent
    leaf_root_dir = project_dir / "leaf_root"
    loader_dir = project_dir / "data_loaders"
    data_dir = project_dir / "data"

    data_dir.mkdir(exist_ok=True)
    (data_dir / "celeba").mkdir(exist_ok=True)
    (data_dir / "femnist").mkdir(exist_ok=True)
    (data_dir / "shakespeare").mkdir(exist_ok=True)

    os.environ["LEAF_ROOT"] = str(leaf_root_dir.resolve())
    os.environ["SAVE_ROOT"] = str(data_dir.resolve())
    os.environ["LOADER_ROOT"] = str(loader_dir.resolve())

    if args["shakespeare"]:
        print("Processing Shakespeare Dataset")

        subprocess.call(str((loader_dir / "shakespeare" / "create_datasets_and_splits.sh").resolve()),
                        stdout=sys.stdout, shell=True, env=os.environ)

        print("Finished processing Shakespeare Dataset")

    if args["femnist"]:
        print("Processing Femnist Dataset")

        subprocess.call(str((loader_dir / "femnist" / "create_datasets_and_splits.sh").resolve()),
                        stdout=sys.stdout, shell=True, env=os.environ)

        print("Finished processing Femnist Dataset")

    if args["celeba"]:
        print("Processing Celeba Dataset")

        download_celeba_dataset()
        subprocess.call(str(( loader_dir / "celeba" / "create_datasets_and_splits.sh").resolve()),
                        stdout=sys.stdout, shell=True, env=os.environ)

        print("Finished processing Celeba Dataset")

    print("Finished downloading all Datasets")
    print("Ending Script Execution")
