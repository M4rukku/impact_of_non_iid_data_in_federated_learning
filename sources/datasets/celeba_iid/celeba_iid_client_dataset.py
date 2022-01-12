from os import PathLike
from sources.datasets.celeba.celeba_client_dataset import CelebaClientDataset


class CelebaIIDClientDataset(CelebaClientDataset):

    def __init__(self,
                 root_data_dir: PathLike,
                 client_identifier: str,
                 subfolder_identifier: str = "celeba_iid"):
        super().__init__(root_data_dir=root_data_dir,
                         subfolder_identifier=subfolder_identifier,
                         client_identifier=client_identifier)
