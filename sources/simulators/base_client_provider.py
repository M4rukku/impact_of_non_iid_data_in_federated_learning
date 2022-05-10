from abc import abstractmethod, ABC

import flwr.client


class BaseClientProvider(ABC):
    def __init__(self,
                 model_template,
                 dataset_factory
                 ):
        self.model_template = model_template
        self.dataset_factory = dataset_factory

    @abstractmethod
    def __call__(self, client_identifier: str) -> flwr.client.Client:
        pass
