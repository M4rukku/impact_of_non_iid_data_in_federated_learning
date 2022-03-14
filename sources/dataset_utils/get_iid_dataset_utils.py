import pickle
from pathlib import Path

from sources.datasets.client_dataset_definitions.client_dataset_processors.client_dataset_processor import \
    ClientDatasetProcessor
from sources.dataset_utils.create_iid_dataset_utils import get_full_iid_dataset_filename, \
    get_default_iid_dataset_filename, get_fractional_iid_dataset_filename, get_globally_shared_iid_dataset_filename
from sources.dataset_utils.dataset import Dataset
from sources.flwr_parameters.exception_definitions import DatasetNotFoundError


def get_data_dir():
    return Path(__file__).parent.parent.parent / "data"


def _get_dataset(dataset_identifier: str, filename: str) -> Dataset:
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


def get_globally_shared_iid_dataset(dataset_identifier: str) -> Dataset:
    return _get_dataset(dataset_identifier,
                        get_globally_shared_iid_dataset_filename(dataset_identifier))


def get_fractional_iid_dataset(dataset_identifier: str, fraction: float) -> Dataset:
    return _get_dataset(dataset_identifier,
                        get_fractional_iid_dataset_filename(dataset_identifier, fraction))


def load_iid_dataset(data_dir: Path, filename: str) -> Dataset:
    with (data_dir / filename).open("rb") as f:
        data = pickle.load(f)

    return Dataset(*data)


def preprocess_dataset(dataset: Dataset, cdp: ClientDatasetProcessor) -> Dataset:
    return Dataset(
        {"x": cdp.process_x(dataset.train["x"]), "y": cdp.process_y(dataset.train["y"])},
        {"x": cdp.process_x(dataset.test["x"]), "y": cdp.process_y(dataset.test["y"])},
        {"x": cdp.process_x(dataset.validation["x"]), "y": cdp.process_y(dataset.validation["y"])}
    )
