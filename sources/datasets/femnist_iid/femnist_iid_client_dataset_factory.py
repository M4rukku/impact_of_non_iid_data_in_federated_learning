from pathlib import Path

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.datasets.femnist_iid.femnist_iid_client_dataset import FemnistIIDClientDataset


class FemnistIIDClientDatasetFactory(ClientDatasetFactory):

    def create_dataset(self, client_identifier: str):
        return FemnistIIDClientDataset(self.root_data_dir, client_identifier)

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(Path(self.root_data_dir) / "femnist_iid")
