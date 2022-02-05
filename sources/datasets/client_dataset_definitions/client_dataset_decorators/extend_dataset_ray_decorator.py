import numpy as np

from sources.datasets.client_dataset_definitions.client_dataset_decorators.base_client_dataset_decorator import BaseClientDatasetDecorator
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset
from sources.ray_tooling.dataset_management_for_ray import fetch_dataset_component_from_ray


class ExtendDatasetRayDecorator(BaseClientDatasetDecorator):

    def __init__(self, client_dataset: ClientDataset, dataset_identifier=None):
        super().__init__(client_dataset)

        self.dataset_identifier = dataset_identifier if dataset_identifier is not None else \
            client_dataset.subfolder_identifier

    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        return np.concatenate((
            fetch_dataset_component_from_ray(self.dataset_identifier, "training_data_x"),
            self.client_dataset.training_data_x
        ))

    @property
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        return np.concatenate((
            fetch_dataset_component_from_ray(self.dataset_identifier, "training_data_y"),
            self.client_dataset.training_data_y
        ))

    @property
    def test_data(self):
        """Returns the Test Data as pair of arrays containing the samples x,
         and classification y"""
        return [self.test_data_x, self.test_data_y]

    @property
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""
        return np.concatenate((
            fetch_dataset_component_from_ray(self.dataset_identifier, "test_data_x"),
            self.client_dataset.test_data_x
        ))

    @property
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        return np.concatenate((
            fetch_dataset_component_from_ray(self.dataset_identifier, "test_data_y"),
            self.client_dataset.test_data_y
        ))

    @property
    def validation_data(self):
        """Returns the Validation Data as pair of arrays containing the
        samples x,
         and classification y"""
        return [self.validation_data_x, self.validation_data_y]

    @property
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""
        return np.concatenate((
            fetch_dataset_component_from_ray(self.dataset_identifier, "validation_data_x"),
            self.client_dataset.validation_data_x
        ))

    @property
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        return np.concatenate((
            fetch_dataset_component_from_ray(self.dataset_identifier, "validation_data_y"),
            self.client_dataset.validation_data_y
        ))
