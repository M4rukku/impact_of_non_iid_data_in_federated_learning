from pathlib import Path

from sources.dataset_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name
from sources.datasets.cifar10_lda.cifar10_lda_client_dataset_processor import \
    Cifar10LdaClientDatasetProcessor
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset
from sources.datasets.client_dataset_definitions.client_dataset_loaders.pickle_file_client_dataset_loader import \
    PickleFileClientDatasetLoader
from sources.datasets.client_dataset_factory_definitions.file_system_based_client_dataset_factory import \
    FileSystemBasedClientDatasetFactory


class Cifar10LdaClientDatasetFactory(FileSystemBasedClientDatasetFactory):

    def __init__(self, root_data_dir: Path, num_partitions: int, concentration: float):
        path_from_data_dir_to_client_dataset = get_lda_cifar10_dataset_name(concentration, num_partitions)
        super().__init__(root_data_dir, path_from_data_dir_to_client_dataset)

    def create_dataset(self, client_identifier: str):
        client_dataset_loader = PickleFileClientDatasetLoader(
            folder_containing_client_datasets=self.path_to_client_datasets
        )

        client_dataset_processor = Cifar10LdaClientDatasetProcessor()

        return ClientDataset(client_identifier=client_identifier,
                             client_dataset_loader=client_dataset_loader,
                             client_dataset_processor=client_dataset_processor)
