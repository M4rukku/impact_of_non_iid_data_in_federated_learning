from abc import ABC, abstractmethod


class ClientDatasetFactory(ABC):

    @abstractmethod
    def create_dataset(self, client_identifier: str):
        pass

    @abstractmethod
    def get_number_of_clients(self):
        pass

