from os import PathLike

from sources.datasets.femnist.femnist_client_dataset import FemnistClientDataset


class FemnistIIDClientDataset(FemnistClientDataset):

    def __init__(self,
                 root_data_dir: PathLike,
                 client_identifier: str,
                 subfolder_identifier: str = "femnist_iid"):

        super().__init__(root_data_dir=root_data_dir,
                         subfolder_identifier=subfolder_identifier,
                         client_identifier=client_identifier)