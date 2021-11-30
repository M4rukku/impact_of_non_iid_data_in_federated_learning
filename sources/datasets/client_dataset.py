from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
import pandas as pd


class ClientDataset(ABC):

    def __init__(self,
                 root_data_dir: PathLike[str],
                 subfolder_identifier: str,
                 client_identifier: str,
                 train_data_filename: str = "train.pickle",
                 test_data_filename: str = "test.pickle",
                 validation_filename: str = "val.pickle"):

        base_data_dir = Path(root_data_dir) / subfolder_identifier
        data_dir = base_data_dir / client_identifier
        self._train_data_filepath = data_dir / train_data_filename
        self._test_data_filepath = data_dir / test_data_filename
        self._validation_data_filepath = data_dir / validation_filename

        self._train_data = None
        self._test_data = None
        self._validation_data = None

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass

    def _lazy_initialise_data(self, data, filepath):
        if data is None:
            data = pd.read_pickle(filepath)
            return self.process_x(data["x"]), self.process_y(data["y"])
        else:
            return data

    @property
    def training_data(self):
        """Returns the Training Data as pair of arrays containing the samples x, and classification y"""
        self._train_data = self._lazy_initialise_data(self._train_data, self._train_data_filepath)
        return self._train_data

    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        self._train_data = self._lazy_initialise_data(self._train_data, self._train_data_filepath)
        return self._train_data[0]

    @property
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        self._train_data = self._lazy_initialise_data(self._train_data, self._train_data_filepath)
        return self._train_data[1]

    @property
    def test_data(self):
        """Returns the Training Data as pair of arrays containing the samples x, and classification y"""
        self._test_data = self._lazy_initialise_data(self._test_data, self._test_data_filepath)
        return self._test_data

    @property
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""
        self._test_data = self._lazy_initialise_data(self._test_data, self._test_data_filepath)
        return self._test_data[0]

    @property
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        self._test_data = self._lazy_initialise_data(self._test_data, self._test_data_filepath)
        return self._test_data[1]

    @property
    def validation_data(self):
        """Returns the Training Data as pair of arrays containing the samples x, and classification y"""
        self._validation_data = self._lazy_initialise_data(self._validation_data, self._validation_data_filepath)
        return self._validation_data

    @property
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""
        self._validation_data = self._lazy_initialise_data(self._validation_data, self._validation_data_filepath)
        return self._validation_data[0]

    @property
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        self._validation_data = self._lazy_initialise_data(self._validation_data, self._validation_data_filepath)
        return self._validation_data[1]