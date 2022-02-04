import pickle
from pathlib import Path
from sources.global_data_properties import DATASET_NAME_LIST
from sources.dataset_utils.create_iid_dataset_utils import get_full_iid_dataset_filename, \
    get_default_iid_dataset_filename, \
    get_fractional_iid_dataset_filename
from sources.dataset_utils.dataset import Dataset
from sources.flwr_parameters.exception_definitions import DatasetNotFoundError


def get_data_dir():
    return Path(__file__).parent.parent.parent / "data"


def _get_dataset(dataset_identifier: str, filename: str) -> Dataset:
    if dataset_identifier not in DATASET_NAME_LIST:
        raise DatasetNotFoundError(f"The Dataset defined by identifier {dataset_identifier} has "
                                   f"not been registered in this simulation framework.")

    data_dir = get_data_dir()
    dataset_filepath = data_dir / filename

    if not dataset_filepath.exists():
        raise DatasetNotFoundError(f"The dataset at path {str(dataset_filepath)} has not been "
                                   f"found. Ensure that it has been initialised with the "
                                   f"create_iid_script first!")

    return load_iid_dataset(data_dir, filename)


def get_full_iid_dataset(dataset_identifier: str) -> Dataset:
    return _get_dataset(dataset_identifier,
                        get_full_iid_dataset_filename(dataset_identifier))


def get_default_iid_dataset(dataset_identifier: str) -> Dataset:
    return _get_dataset(dataset_identifier,
                        get_default_iid_dataset_filename(dataset_identifier))


def get_fractional_iid_dataset(dataset_identifier: str, fraction: float) -> Dataset:
    return _get_dataset(dataset_identifier,
                        get_fractional_iid_dataset_filename(dataset_identifier, fraction))


def load_iid_dataset(data_dir: Path, filename: str) -> Dataset:
    with (data_dir / filename).open("rb") as f:
        data = pickle.load(f)

    return Dataset(*data)
