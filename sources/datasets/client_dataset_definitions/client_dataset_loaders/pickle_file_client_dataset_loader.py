import logging
import pickle
from pathlib import Path
from sources.dataset_utils.unmodifyable_attributes_trait import UnmodifiableAttributes
from sources.datasets.client_dataset_definitions.client_dataset_loaders.client_dataset_loader import ClientDatasetLoader, DatasetComponents, \
    MinimalDataset


class PickleFileClientDatasetLoader(ClientDatasetLoader, UnmodifiableAttributes):

    def __init__(self,
                 folder_containing_client_datasets: Path,
                 train_data_filename: str = "train.pickle",
                 test_data_filename: str = "test.pickle",
                 validation_filename: str = "val.pickle"
                 ):
        self._folder_containing_client_datasets = folder_containing_client_datasets
        self._train_data_filename = train_data_filename
        self._test_data_filename = test_data_filename
        self._validation_data_filename = validation_filename

    def load_dataset(self,
                     client_identifier: str,
                     dataset_component: DatasetComponents
                     ) -> MinimalDataset:

        dataset_path = self._folder_containing_client_datasets / client_identifier

        if dataset_component == DatasetComponents.TRAIN:
            dataset_path = dataset_path / self._train_data_filename
        elif dataset_component == DatasetComponents.TEST:
            dataset_path = dataset_path / self._test_data_filename
        elif dataset_component == DatasetComponents.VALIDATION:
            dataset_path = dataset_path / self._validation_data_filename
        else:
            raise RuntimeError(f"Loading Dataset is impossible since case {dataset_component} "
                               f"has not been implemented in PickleFileClientDatasetLoader")
        try:
            with dataset_path.open("rb") as f:
                dataset = pickle.load(f)
        except BaseException as e:
            logging.error(f"Failed to load the following dataset: {dataset_path}")
            logging.error(e.__repr__())

        return dataset
