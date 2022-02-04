from sources.datasets.shakespeare.shakespeare_client_dataset import ShakespeareClientDataset
from sources.global_data_properties import LEAF_CHARACTERS
from os import PathLike
from os import PathLike


class ShakespeareIIDClientDataset(ShakespeareClientDataset):

    def __init__(self,
                 root_data_dir: PathLike,
                 client_identifier: str,
                 subfolder_identifier="shakespeare_iid"):
        super().__init__(root_data_dir=root_data_dir,
                         client_identifier=client_identifier,
                         subfolder_identifier=subfolder_identifier)
