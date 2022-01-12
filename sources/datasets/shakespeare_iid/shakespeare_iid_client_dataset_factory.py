from pathlib import Path

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.datasets.shakespeare_iid.shakespeare_iid_client_dataset import \
    ShakespeareIIDClientDataset
from sources.global_data_properties import LEAF_CHARACTERS


class ShakespeareIIDClientDatasetFactory(ClientDatasetFactory):

    def create_dataset(self, client_identifier: str, alphabet: str = LEAF_CHARACTERS):
        return ShakespeareIIDClientDataset(self.root_data_dir, client_identifier, alphabet)

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(Path(self.root_data_dir) / "shakespeare_iid")
