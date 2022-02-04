from os import PathLike
import numpy as np

from sources.datasets.client_dataset import ClientDataset
from sources.datasets.client_dataset_processor import ClientDatasetProcessor
from sources.datasets.femnist.femnist_client_dataset_processor import FemnistClientDatasetProcessor


class FemnistClientDataset(ClientDataset):

    def __init__(self,
                 root_data_dir: PathLike,
                 client_identifier: str,
                 subfolder_identifier: str = "femnist",
                 client_dataset_processor: ClientDatasetProcessor = FemnistClientDatasetProcessor()
                 ):
        super().__init__(root_data_dir=root_data_dir,
                         subfolder_identifier=subfolder_identifier,
                         client_identifier=client_identifier,
                         client_dataset_processor=client_dataset_processor)
