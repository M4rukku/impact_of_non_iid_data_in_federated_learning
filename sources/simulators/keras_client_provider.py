import logging

from sources.flwr.flwr_clients.keras_client import KerasClient
from sources.metrics.default_metrics_tf import DEFAULT_METRICS
from sources.simulators.base_client_provider import BaseClientProvider


class KerasClientProvider(BaseClientProvider):
    def __init__(self,
                 model_template,
                 dataset_factory,
                 metrics=DEFAULT_METRICS,
                 fitting_callbacks=None,
                 evaluation_callbacks=None):
        super().__init__(model_template, dataset_factory)
        self.metrics = metrics
        self.fitting_callbacks = fitting_callbacks
        self.evaluation_callbacks = evaluation_callbacks

    def __call__(self, client_identifier: str):
        try:
            base_client = KerasClient(self.model_template,
                                      self.dataset_factory.create_dataset(client_identifier),
                                      self.metrics,
                                      self.fitting_callbacks,
                                      self.evaluation_callbacks)
        except Exception as e:
            logging.error(str(e))
            base_client = None
        return base_client
