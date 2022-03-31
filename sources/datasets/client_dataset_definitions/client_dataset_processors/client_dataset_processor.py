from abc import ABC, abstractmethod

from sources.utils.unmodifyable_attributes_trait import UnmodifiableAttributes


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
