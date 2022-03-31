import functools
import logging
import typing
import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common import Config, Properties

from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset
from sources.utils.client_parameters import \
    FederatedEvaluationParameters, FittingParameters
from sources.utils.exception_definitions import ConfigContainsUnknownPropertyError
from sources.metrics.default_metrics import DEFAULT_METRICS
from sources.models.model_template import ModelTemplate


def initialise_model(self, parameters=None):
    self.model = self.model_template.get_model()
    if parameters is not None:
        self.model.set_weights(parameters)
    self.optimizer = self.model_template.get_optimizer(model=self.model)
    self.loss = self.model_template.get_loss(model=self.model)
    self.model.compile(self.optimizer, self.loss, self.metrics)
    self.model_initialised = True


def lazy_client_initializer(func):
    @functools.wraps(func)
    def wrapper_decorator(self, *args, **kwargs):
        if not self.model_initialised:
            initialise_model(self)

        value = func(self, *args, **kwargs)
        return value

    return wrapper_decorator


class KerasClient(fl.client.NumPyClient):

    def __init__(self,
                 model_template: ModelTemplate,
                 dataset: ClientDataset,
                 metrics=DEFAULT_METRICS,
                 fitting_callbacks: list[tf.keras.callbacks.Callback] = None,
                 evaluation_callbacks: list[tf.keras.callbacks.Callback] = None
                 ):
        # set_global_determinism(DEFAULT_SEED)
        tf.keras.backend.clear_session()
        self.model_template: ModelTemplate = model_template
        self.dataset = dataset
        self.fitting_callbacks = fitting_callbacks
        self.evaluation_callbacks = evaluation_callbacks

        self.model_initialised = False
        self.model: tf.keras.Model = None
        self.optimizer: tf.keras.optimizers.Optimizer = None
        self.loss: tf.keras.losses.Loss = None
        self.metrics: typing.List[typing.Union[tf.keras.metrics.Metric, str]] = metrics

    @lazy_client_initializer
    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    @lazy_client_initializer
    def get_properties(self, config: Config) -> Properties:
        if len(config.items()) > 0:
            raise ConfigContainsUnknownPropertyError(f"""
            The configuration in BaseClient wants information about the 
            following keys {config.keys()} which are not all supported. Please 
            add support for unimplemented property data.
            """)
        return config

    def fit(self, parameters: np.array, config: FittingParameters):
        """Fit model and return new weights as well as number of training
        examples."""
        initialise_model(self, parameters)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        try:
            with self.dataset:
                history = self.model.fit(
                    self.dataset.training_data_x,
                    self.dataset.training_data_y,
                    batch_size,
                    epochs,
                    shuffle=True,
                    validation_data=self.dataset.test_data,
                    callbacks=self.fitting_callbacks,
                    verbose=1
                )
                num_examples_train = len(self.dataset.training_data_x)
        except Exception as e:
            logging.error(str(e))

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        return (parameters_prime,
                num_examples_train,
                {key: entry[0] for key, entry in history.history.items()})

    def evaluate(self, parameters, config: FederatedEvaluationParameters):
        batch_size: int = config["batch_size"]
        val_steps: int = config["val_steps"]

        initialise_model(self, parameters)

        with self.dataset:
            data_x, data_y = np.array(self.dataset.validation_data_x), np.array(
                self.dataset.validation_data_y)
            rng = np.random.default_rng()
            indices = rng.choice(len(data_x), len(data_x), replace=False)

            result_dict = \
                self.model.evaluate(data_x[indices],
                                    data_y[indices],
                                    batch_size=batch_size,
                                    steps=val_steps,
                                    return_dict=True,
                                    callbacks=self.evaluation_callbacks)

            data_len = len(self.dataset.validation_data_x)

        result = (result_dict["loss"],
                  int(min(data_len, batch_size *
                          (val_steps if val_steps is not None
                           else data_len / batch_size))), result_dict)
        return result
