from pathlib import Path

from sources.datasets.celeba_iid.celeba_iid_client_dataset import CelebaIIDClientDataset
from sources.datasets.client_dataset_factory import ClientDatasetFactory


class CelebaIIDClientDatasetFactory(ClientDatasetFactory):

    def create_dataset(self, client_identifier: str):
        return CelebaIIDClientDataset(self.root_data_dir, client_identifier)

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(Path(self.root_data_dir) / "celeba_iid")
