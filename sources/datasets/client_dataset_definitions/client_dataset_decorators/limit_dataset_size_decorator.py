from typing import Tuple, Optional
from sources.datasets.client_dataset_definitions.client_dataset_decorators.base_client_dataset_decorator import \
    BaseClientDatasetDecorator
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset


class LimitDatasetSizeDecorator(BaseClientDatasetDecorator):
    # Since this class creates deterministic datasets (as in global data will always be before client data,
    # need to ensure that we shuffle the data during fitting (see BaseClient)

    def __init__(self, client_dataset: ClientDataset,
                 dataset_size_limiters: Tuple[Optional[int], Optional[int], Optional[int]],
                 ):
        super().__init__(client_dataset)

        self.dataset_size_limiters_train = dataset_size_limiters[0]
        self.dataset_size_limiters_test = dataset_size_limiters[1]
        self.dataset_size_limiters_validate = dataset_size_limiters[2]


    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        training_data_x = self.client_dataset.training_data_x
        dataset_length = len(training_data_x)

        if (self.dataset_size_limiters_train is not None and
                self.dataset_size_limiters_train < dataset_length):
            return training_data_x[:self.dataset_size_limiters_train]
        else:
            return training_data_x

    @property
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        training_data_y = self.client_dataset.training_data_y
        dataset_length = len(training_data_y)

        if (self.dataset_size_limiters_train is not None and
                self.dataset_size_limiters_train < dataset_length):
            return training_data_y[:self.dataset_size_limiters_train]
        else:
            return training_data_y

    @property
    def test_data(self):
        """Returns the Test Data as pair of arrays containing the samples x,
         and classification y"""
        return [self.test_data_x, self.test_data_y]

    @property
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""
        test_data_x = self.client_dataset.test_data_x
        dataset_length = len(test_data_x)

        if (self.dataset_size_limiters_test is not None and
                self.dataset_size_limiters_test < dataset_length):
            return test_data_x[:self.dataset_size_limiters_test]
        else:
            return test_data_x

    @property
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        test_data_y = self.client_dataset.test_data_y
        dataset_length = len(test_data_y)

        if (self.dataset_size_limiters_test is not None and
                self.dataset_size_limiters_test < dataset_length):
            return test_data_y[:self.dataset_size_limiters_test]
        else:
            return test_data_y

    @property
    def validation_data(self):
        """Returns the Validation Data as pair of arrays containing the
        samples x,
         and classification y"""
        return [self.validation_data_x, self.validation_data_y]

    @property
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""

        validation_data_x = self.client_dataset.validation_data_x
        dataset_length = len(validation_data_x)

        if (self.dataset_size_limiters_validate is not None and
                self.dataset_size_limiters_validate < dataset_length):
            return validation_data_x[:self.dataset_size_limiters_validate]
        else:
            return validation_data_x

    @property
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        validation_data_y = self.client_dataset.validation_data_y
        dataset_length = len(validation_data_y)

        if (self.dataset_size_limiters_validate is not None and
                self.dataset_size_limiters_validate < dataset_length):
            return validation_data_y[:self.dataset_size_limiters_validate]
        else:
            return validation_data_y
