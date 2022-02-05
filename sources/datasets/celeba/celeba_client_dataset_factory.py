from pathlib import Path

from sources.datasets.celeba.celeba_client_dataset_processor import CelebaClientDatasetProcessor
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset
from sources.datasets.client_dataset_definitions.client_dataset_loaders.pickle_file_client_dataset_loader import \
    PickleFileClientDatasetLoader
from sources.datasets.client_dataset_factory_definitions.file_system_based_client_dataset_factory import \
    FileSystemBasedClientDatasetFactory


class CelebaClientDatasetFactory(FileSystemBasedClientDatasetFactory):

    def __init__(self, root_data_dir: Path, path_from_data_dir_to_client_dataset="celeba"):
        super().__init__(root_data_dir, path_from_data_dir_to_client_dataset)

    def create_dataset(self, client_identifier: str):
        client_dataset_loader = PickleFileClientDatasetLoader(
            folder_containing_client_datasets=self.path_to_client_datasets
        )

        client_dataset_processor = CelebaClientDatasetProcessor()

        return ClientDataset(client_identifier=client_identifier,
                             client_dataset_loader=client_dataset_loader,
                             client_dataset_processor=client_dataset_processor)
