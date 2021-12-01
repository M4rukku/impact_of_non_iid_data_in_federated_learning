from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path


class ClientDatasetFactory(ABC):

    def __init__(self, root_data_dir: PathLike[str]):
        self.root_data_dir = root_data_dir

    @abstractmethod
    def create_dataset(self, client_identifier: str):
        pass

    @abstractmethod
    def get_number_of_clients(self):
        pass

    @staticmethod
    def _number_of_files_in_dir(dir_name: PathLike[str]):
        dir_path = Path(dir_name)
        assert dir_path.is_dir()
        return len(list(dir_path.iterdir()))
