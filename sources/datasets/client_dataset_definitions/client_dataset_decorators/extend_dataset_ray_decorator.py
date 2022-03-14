from typing import Tuple

import numpy as np

from sources.datasets.client_dataset_definitions.client_dataset_decorators.base_client_dataset_decorator import \
    BaseClientDatasetDecorator
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset
from sources.ray_tooling.dataset_management_for_ray import fetch_dataset_component_from_ray


class ExtendDatasetRayDecorator(BaseClientDatasetDecorator):
    # Since this class creates deterministic datasets (as in global data will always be before client data,
    # need to ensure that we shuffle the data during fitting (see BaseClient)

    def __init__(self, client_dataset: ClientDataset,
                 dataset_identifier,
                 subset_size: Tuple[int, int, int],
                 shared_dataset_size: Tuple[int, int, int]  # Train, Test, Validate
                 ):
        super().__init__(client_dataset)

        self.dataset_identifier = dataset_identifier
        self.subset_size = subset_size
        self.shared_dataset_size = shared_dataset_size

        rng = np.random.default_rng()
        self.selection_train = rng.choice(shared_dataset_size[0], subset_size[0], replace=False)
        self.selection_test = rng.choice(shared_dataset_size[1], subset_size[1], replace=False)
        self.selection_val = rng.choice(shared_dataset_size[2], subset_size[2], replace=False)

    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        if self.subset_size[0]>0:
            data_x = fetch_dataset_component_from_ray(self.dataset_identifier, "training_data_x")
            data_x = data_x[self.selection_train]
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
            data_y = fetch_dataset_component_from_ray(self.dataset_identifier, "training_data_y")
            data_y = data_y[self.selection_train]
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
            data_x = fetch_dataset_component_from_ray(self.dataset_identifier, "test_data_x")
            data_x = data_x[self.selection_test]
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
            data_y = fetch_dataset_component_from_ray(self.dataset_identifier, "test_data_y")
            data_y = data_y[self.selection_test]
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
            data_x = fetch_dataset_component_from_ray(self.dataset_identifier, "validation_data_x")
            data_x = data_x[self.selection_val]
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
            data_y = fetch_dataset_component_from_ray(self.dataset_identifier, "validation_data_y")
            data_y = data_y[self.selection_val]
            return np.concatenate((
                data_y,
                self.client_dataset.validation_data_y
            ))
        else:
            return self.client_dataset.validation_data_y