from pathlib import Path

from sources.datasets.client_dataset_factory import ClientDatasetFactory
from sources.datasets.shakespeare.shakespeare_client_dataset import ShakespeareClientDataset
from global_data_properties import LEAF_CHARACTERS


class ShakespeareClientDatasetFactory(ClientDatasetFactory):

    def create_dataset(self, client_identifier: str, alphabet: str = LEAF_CHARACTERS):
        return ShakespeareClientDataset(self.root_data_dir, client_identifier, alphabet)

    def get_number_of_clients(self):
        return self._number_of_files_in_dir(Path(self.root_data_dir) / "shakespeare")
