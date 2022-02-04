from sources.datasets.client_dataset import ClientDataset


class BaseClientDatasetDecorator(ClientDataset):
    # noinspection PyMissingConstructor
    def __init__(self,
                 client_dataset: ClientDataset):
        self.client_dataset = client_dataset

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features
         before being fed to the model."""
        return self.client_dataset.process_x(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return self.client_dataset.process_x(raw_y_batch)

    @property
    def training_data(self):
        """Returns the Training Data as pair of arrays containing the samples x,
         and classification y"""
        return self.client_dataset.training_data

    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        return self.client_dataset.training_data_x

    @property
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        return self.client_dataset.training_data_y

    @property
    def test_data(self):
        """Returns the Training Data as pair of arrays containing the samples x,
         and classification y"""
        return self.client_dataset.test_data

    @property
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""
        return self.client_dataset.test_data_x

    @property
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        return self.client_dataset.test_data_y

    @property
    def validation_data(self):
        """Returns the Validation Data as pair of arrays containing the
        samples x,
         and classification y"""
        return self.client_dataset.validation_data

    @property
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""
        return self.client_dataset.validation_data_x

    @property
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        return self.client_dataset.validation_data_y

    def __enter__(self):
        self.client_dataset.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client_dataset.__exit__(exc_type, exc_value, exc_traceback)
