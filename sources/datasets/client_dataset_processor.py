from abc import ABC, abstractmethod


class UnmodifiableAttributes(object):
    '''
    Ensures that inheritng objects will not be able to modify their variables -- this allows us
    to treat the Dataset Processors as quasi immutable objects (They may allocate state,
    but it will be constant over all instances given the same constructor parameters)
    '''
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise RuntimeError("Can't modify immutable object's attribute: {}".format(key))


class ClientDatasetProcessor(ABC, UnmodifiableAttributes):
    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features
         before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class IdentityClientDatasetProcessor(ClientDatasetProcessor):
    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features
         before being fed to the model."""
        return raw_x_batch

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return raw_y_batch
