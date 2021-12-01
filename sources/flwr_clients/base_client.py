import random
import typing
import flwr as fl
import numpy as np
import tensorflow as tf

from sources.datasets.client_dataset import ClientDataset
from sources.flwr_parameters.client_parameters import FederatedEvaluationParameters, FittingParameters
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate


class BaseClient(fl.client.NumPyClient):

    def __init__(self,
                 model_template: ModelTemplate,
                 dataset: ClientDataset,
                 metrics=DEFAULT_METRICS,
                 fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
                 evaluation_callbacks: list[tf.keras.callbacks.Callback] = None
                 ):
        self.model: tf.keras.Model = model_template.get_model()
        self.optimizer: tf.keras.optimizers.Optimizer = model_template.get_optimizer()
        self.loss: tf.keras.losses.Loss = model_template.get_loss()
        self.metrics: typing.List[typing.Union[tf.keras.metrics.Metric, str]] = metrics

        self.model.compile(self.optimizer, self.loss, self.metrics)
        self.dataset = dataset

        self.fitting_callbacks = fitting_callbacks
        self.evaluation_callbacks = evaluation_callbacks

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters: np.array, config: FittingParameters):
        """Fit model and return new weights as well as number of training
        examples."""

        # Load model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.dataset.test_data_x,
            self.dataset.test_data_y,
            batch_size,
            epochs,
            validation_data=self.dataset.validation_data,
            callbacks=self.fitting_callbacks,
            verbose=1
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.dataset.test_data_x)

        return parameters_prime, num_examples_train, {key: entry[0] for key, entry in history.history.items()}

    def evaluate(self, parameters, config: FederatedEvaluationParameters):
        rand = random.randint(1, 100)
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        val_steps: int = config["val_steps"]

        """Evaluate using provided parameters."""
        self.model.set_weights(parameters)
        result_dict = self.model.evaluate(self.dataset.validation_data_x, self.dataset.validation_data_y,
                                          batch_size=batch_size, steps=val_steps, return_dict=True,
                                          callbacks=self.evaluation_callbacks)

        data_len = len(self.dataset.validation_data_x)
        return (result_dict["loss"], min(data_len, batch_size * (val_steps if val_steps is not None
                                                                 else data_len / batch_size)), result_dict)
