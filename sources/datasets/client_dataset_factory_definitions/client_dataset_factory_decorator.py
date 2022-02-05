from typing import Type, Dict, Any

from sources.datasets.client_dataset_definitions.client_dataset_decorators.base_client_dataset_decorator import BaseClientDatasetDecorator
from sources.datasets.client_dataset_factory_definitions.client_dataset_factory import ClientDatasetFactory


# noinspection PyMissingConstructor
class DecoratedClientDatasetFactoryDecorator(ClientDatasetFactory):
    '''
    This class decorates a client dataset factory and applies the decorator type with kwargs to
    the class created by the dataset_factory
    '''

    def __init__(self,
                 dataset_factory: ClientDatasetFactory,
                 client_dataset_decorator: Type[BaseClientDatasetDecorator],
                 client_dataset_decorator_kwargs: Dict[str, Any]
                 ):
        self.decorated_dataset_factory = dataset_factory
        self.client_dataset_decorator = client_dataset_decorator
        self.client_dataset_decorator_kwargs = client_dataset_decorator_kwargs

    def create_dataset(self, client_identifier: str):
        dataset = self.decorated_dataset_factory.create_dataset(client_identifier)
        kwargs = {
            "client_dataset": dataset,
            **self.client_dataset_decorator_kwargs
        }
        return self.client_dataset_decorator(
            **kwargs
        )

    def get_number_of_clients(self):
        self.decorated_dataset_factory.get_number_of_clients()
