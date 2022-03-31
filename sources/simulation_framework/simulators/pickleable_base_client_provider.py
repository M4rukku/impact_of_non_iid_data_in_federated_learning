import logging

from sources.flwr_clients.base_client import BaseClient


class PickleableBaseClientProvider:
    def __init__(self, model_template, dataset_factory, metrics,
                 fitting_callbacks, evaluation_callbacks):
        self.model_template = model_template
        self.dataset_factory = dataset_factory
        self.metrics = metrics
        self.fitting_callbacks = fitting_callbacks
        self.evaluation_callbacks = evaluation_callbacks

    def __call__(self, client_identifier: str):
        try:
            base_client = BaseClient(self.model_template,
                                     self.dataset_factory.create_dataset(client_identifier),
                                     self.metrics,
                                     self.fitting_callbacks,
                                     self.evaluation_callbacks)
        except Exception as e:
            logging.error(str(e))
            base_client = None
        return base_client