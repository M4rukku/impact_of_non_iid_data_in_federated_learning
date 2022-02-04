import functools
import gc
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
import pandas as pd

from sources.datasets.client_dataset_processor import ClientDatasetProcessor
from sources.flwr_parameters.exception_definitions import OutsideOfContextError


def throw_error_outside_context(func):
    @functools.wraps(func)
    def wrapper_decorator(self, *args, **kwargs):
        if not self.within_context:
            raise OutsideOfContextError(
                """Error: Tried to access client Dataset outside of context 
                manager. This might lead to data leaks and bad use of 
                memory. Please wrap the usage of ClientDataset.dataset_x 
                inside a "with statement". """)
        else:
            value = func(self, *args, **kwargs)
            return value

    return wrapper_decorator


class ClientDataset(ABC):

    def __init__(self,
                 root_data_dir: PathLike,
                 subfolder_identifier: str,
                 client_identifier: str,
                 client_dataset_processor: ClientDatasetProcessor,
                 train_data_filename: str = "train.pickle",
                 test_data_filename: str = "test.pickle",
                 validation_filename: str = "val.pickle"
                 ):

        base_data_dir = Path(root_data_dir) / subfolder_identifier
        data_dir = base_data_dir / client_identifier

        self.subfolder_identifier = subfolder_identifier
        self._train_data_filepath = data_dir / train_data_filename
        self._test_data_filepath = data_dir / test_data_filename
        self._validation_data_filepath = data_dir / validation_filename

        self.client_dataset_processor = client_dataset_processor
        self._train_data = None
        self._test_data = None
        self._validation_data = None
        self.within_context = False

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features
         before being fed to the model."""
        return self.client_dataset_processor.process_x(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return self.client_dataset_processor.process_y(raw_y_batch)

    def _lazy_initialise_data(self, data, filepath):
        if data is None:
            data = pd.read_pickle(filepath)
            return self.process_x(data["x"]), self.process_y(data["y"])
        else:
            return data

    @property
    @throw_error_outside_context
    def training_data(self):
        """Returns the Training Data as pair of arrays containing the samples x,
         and classification y"""
        self._train_data = self._lazy_initialise_data(self._train_data,
                                                      self._train_data_filepath)
        return self._train_data

    @property
    @throw_error_outside_context
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        self._train_data = self._lazy_initialise_data(self._train_data,
                                                      self._train_data_filepath)
        return self._train_data[0]

    @property
    @throw_error_outside_context
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        self._train_data = self._lazy_initialise_data(self._train_data,
                                                      self._train_data_filepath)
        return self._train_data[1]

    @property
    @throw_error_outside_context
    def test_data(self):
        """Returns the Training Data as pair of arrays containing the samples x,
         and classification y"""
        self._test_data = self._lazy_initialise_data(self._test_data,
                                                     self._test_data_filepath)
        return self._test_data

    @property
    @throw_error_outside_context
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""
        self._test_data = self._lazy_initialise_data(self._test_data,
                                                     self._test_data_filepath)
        return self._test_data[0]

    @property
    @throw_error_outside_context
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        self._test_data = self._lazy_initialise_data(self._test_data,
                                                     self._test_data_filepath)
        return self._test_data[1]

    @property
    @throw_error_outside_context
    def validation_data(self):
        """Returns the Validation Data as pair of arrays containing the
        samples x,
         and classification y"""
        self._validation_data = self._lazy_initialise_data(
            self._validation_data, self._validation_data_filepath)
        return self._validation_data

    @property
    @throw_error_outside_context
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""
        self._validation_data = self._lazy_initialise_data(
            self._validation_data, self._validation_data_filepath)
        return self._validation_data[0]

    @property
    @throw_error_outside_context
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        self._validation_data = self._lazy_initialise_data(
            self._validation_data, self._validation_data_filepath)
        return self._validation_data[1]

    def __enter__(self):
        self.within_context = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.within_context = False
        self._train_data = None
        self._test_data = None
        self._validation_data = None
        gc.collect()
