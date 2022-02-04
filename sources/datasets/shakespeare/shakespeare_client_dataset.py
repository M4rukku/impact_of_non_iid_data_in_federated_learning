from os import PathLike
from typing import List
import tensorflow as tf
import numpy as np

from sources.datasets.client_dataset import ClientDataset
from sources.datasets.client_dataset_processor import ClientDatasetProcessor
from sources.datasets.shakespeare.shakespeare_client_dataset_processor import \
    ShakespeareClientDatasetProcessor
from sources.global_data_properties import LEAF_CHARACTERS


class ShakespeareClientDataset(ClientDataset):
    def __init__(self,
                 root_data_dir: PathLike,
                 client_identifier: str,
                 subfolder_identifier="shakespeare",
                 client_dataset_processor: ClientDatasetProcessor =
                 ShakespeareClientDatasetProcessor(alphabet=LEAF_CHARACTERS),
                 ):
        super().__init__(root_data_dir=root_data_dir,
                         subfolder_identifier=subfolder_identifier,
                         client_identifier=client_identifier,
                         client_dataset_processor=client_dataset_processor)
