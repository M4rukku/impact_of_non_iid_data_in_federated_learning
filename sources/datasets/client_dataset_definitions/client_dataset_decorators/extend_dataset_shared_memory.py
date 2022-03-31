from typing import Tuple

import numpy as np

from sources.utils.dataset import Dataset
from sources.datasets.client_dataset_definitions.client_dataset_decorators.base_client_dataset_decorator import \
    BaseClientDatasetDecorator
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset


class ExtendDatasetSharedMemory(BaseClientDatasetDecorator):
    # Since this class creates deterministic datasets (as in global data will always be before client data,
    # need to ensure that we shuffle the data during fitting (see BaseClient)

    def __init__(self, client_dataset: ClientDataset,
                 subset_size: Tuple[int, int, int],
                 shared_dataset: Dataset
                 ):
        super().__init__(client_dataset)

        self.subset_size = subset_size
        self.shared_dataset = shared_dataset
        self.shared_dataset_size = (len(shared_dataset.train["x"]),
                                    len(shared_dataset.test["x"]),
                                    len(shared_dataset.validation["x"]))

        rng = np.random.default_rng()
        self.selection_train = rng.choice(self.shared_dataset_size[0], subset_size[0],
                                          replace=False)
        self.selection_test = rng.choice(self.shared_dataset_size[1], subset_size[1], replace=False)
        self.selection_val = rng.choice(self.shared_dataset_size[2], subset_size[2], replace=False)

    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        if self.subset_size[0] > 0:
            data_x = self.shared_dataset.train["x"][self.selection_train]
            return np.concatenate((
                data_x,
                self.client_dataset.training_data_x
            ))
        else:
            return self.client_dataset.training_data_x

    @property
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        if self.subset_size[0] > 0:
            data_y = self.shared_dataset.train["y"][self.selection_train]
            return np.concatenate((
                data_y,
                self.client_dataset.training_data_y
            ))
        else:
            return self.client_dataset.training_data_y

    @property
    def test_data(self):
        """Returns the Test Data as pair of arrays containing the samples x,
         and classification y"""
        return [self.test_data_x, self.test_data_y]

    @property
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""
        if self.subset_size[1] > 0:
            data_x = self.shared_dataset.test["x"][self.selection_test]
            return np.concatenate((
                data_x,
                self.client_dataset.test_data_x
            ))
        else:
            return self.client_dataset.test_data_x

    @property
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        if self.subset_size[1] > 0:
            data_y = self.shared_dataset.train["y"][self.selection_test]
            return np.concatenate((
                data_y,
                self.client_dataset.test_data_y
            ))
        else:
            return self.client_dataset.test_data_y

    @property
    def validation_data(self):
        """Returns the Validation Data as pair of arrays containing the
        samples x,
         and classification y"""
        return [self.validation_data_x, self.validation_data_y]

    @property
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""
        if self.subset_size[2] > 0:
            data_x = self.shared_dataset.validation["x"][self.selection_val]
            return np.concatenate((
                data_x,
                self.client_dataset.validation_data_x
            ))
        else:
            return self.client_dataset.validation_data_x

    @property
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        if self.subset_size[0] > 0:
            data_y = self.shared_dataset.validation["y"][self.selection_val]
            return np.concatenate((
                data_y,
                self.client_dataset.validation_data_y
            ))
        else:
            return self.client_dataset.validation_data_y
