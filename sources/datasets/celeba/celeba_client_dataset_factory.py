from pathlib import Path

from sources.datasets.celeba.celeba_client_dataset import CelebaClientDataset
from sources.datasets.client_dataset_factory import ClientDatasetFactory


class CelebaClientDatasetFactory(ClientDatasetFactory):

    def create_dataset(self, client_identifier: str):
        return CelebaClientDataset(self.root_data_dir, client_identifier)

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(Path(self.root_data_dir) / "celeba")
