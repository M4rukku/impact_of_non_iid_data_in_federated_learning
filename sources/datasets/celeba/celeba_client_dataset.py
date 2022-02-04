from os import PathLike
import numpy as np

from sources.datasets.celeba.celeba_client_dataset_processor import CelebaClientDatasetProcessor
from sources.datasets.client_dataset import ClientDataset
from sources.datasets.client_dataset_processor import ClientDatasetProcessor


class CelebaClientDataset(ClientDataset):

    def __init__(self,
                 root_data_dir: PathLike,
                 client_identifier: str,
                 subfolder_identifier: str = "celeba",
                 client_dataset_processor: ClientDatasetProcessor = CelebaClientDatasetProcessor()
                 ):
        super().__init__(root_data_dir=root_data_dir,
                         subfolder_identifier=subfolder_identifier,
                         client_identifier=client_identifier,
                         client_dataset_processor=client_dataset_processor)
