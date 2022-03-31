from abc import abstractmethod, ABC

import flwr.client


class BaseClientProvider(ABC):
    def __init__(self,
                 model_template,
                 dataset_factory,
                 metrics):
        self.model_template = model_template
        self.dataset_factory = dataset_factory
        self.metrics = metrics

    @abstractmethod
    def __call__(self, *args, **kwargs) -> flwr.client.Client:
        pass
