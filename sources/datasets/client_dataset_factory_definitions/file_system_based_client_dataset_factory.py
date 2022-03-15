from abc import abstractmethod
from pathlib import Path

from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import \
    ClientDatasetFactory


class FileSystemBasedClientDatasetFactory(ClientDatasetFactory):

    def __init__(self,
                 root_data_dir: Path,
                 path_from_data_dir_to_client_dataset: str
                 ):
        self.root_data_dir = root_data_dir
        self.path_from_data_dir_to_client_dataset = path_from_data_dir_to_client_dataset

        self.path_to_client_datasets = \
            self.root_data_dir / self.path_from_data_dir_to_client_dataset

    @abstractmethod
    def create_dataset(self, client_identifier: str):
        pass

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(self.path_to_client_datasets)

    @staticmethod
    def _number_of_files_in_dir(dir_path: Path):
        assert dir_path.is_dir()
        return len(list(filter(lambda f: f.is_dir(), dir_path.iterdir())))
