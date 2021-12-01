from pathlib import Path

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.datasets.femnist.femnist_client_dataset import FemnistClientDataset


class FemnistClientDatasetFactory(ClientDatasetFactory):

    def create_dataset(self, client_identifier: str):
        return FemnistClientDataset(self.root_data_dir, client_identifier)

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(Path(self.root_data_dir) / "femnist")
