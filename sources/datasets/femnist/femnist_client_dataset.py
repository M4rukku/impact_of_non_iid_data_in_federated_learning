from os import PathLike
import numpy as np

from sources.datasets.client_dataset import ClientDataset


class FemnistClientDataset(ClientDataset):

    def __init__(self, root_data_dir: PathLike, client_identifier: str):

        super().__init__(root_data_dir=root_data_dir,
                         subfolder_identifier="femnist",
                         client_identifier=client_identifier)

    def process_x(self, raw_x_batch):
        return np.array([np.array(img) for img in raw_x_batch])

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
