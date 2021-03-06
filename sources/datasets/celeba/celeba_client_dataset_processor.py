import numpy as np

from sources.datasets.client_dataset_definitions.client_dataset_processors.client_dataset_processor import ClientDatasetProcessor


class CelebaClientDatasetProcessor(ClientDatasetProcessor):

    def process_x(self, raw_x_batch):
        x_batch = [np.array(img) / 255.0 - 0.5 for img in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(list(map(int, raw_y_batch)))
