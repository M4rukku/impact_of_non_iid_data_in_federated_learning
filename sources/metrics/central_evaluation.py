import logging
from typing import Optional, Tuple, Dict, Union, List, Callable
import numpy
import tensorflow as tf

from sources.dataset_utils.create_iid_dataset_utils import Dataset
from sources.datasets.client_dataset import ClientDataset
from sources.flwr_parameters.set_random_seeds import set_global_determinism
from sources.models.make_keras_pickleable import make_keras_pickleable
from sources.models.model_template import ModelTemplate

EvalFunType = Callable[[List[numpy.ndarray]],
                       Optional[Tuple[float, Dict[str, Union[bool, bytes, float, int, str]]]]]


class PickleableEvaluationFunction:
    def __init__(self, model, optimizer, loss, metrics, evaluation_x_data, evaluation_y_data):
        make_keras_pickleable()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.evaluation_x_data = evaluation_x_data
        self.evaluation_y_data = evaluation_y_data

    def __call__(self, results: List[numpy.ndarray]) -> \
            Optional[Tuple[float, Dict[str, Union[bool, bytes, float, int, str]]]]:
        set_global_determinism()
        logging.warning("Starting Centralised Evaluation")

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        self.model.set_weights(results)
        results = self.model.evaluate(x=self.evaluation_x_data, y=self.evaluation_y_data,
                                      return_dict=True)
        logging.warning(f"Finished Centralised Evaluation with result {results.items()}")
        return results["loss"], results


def create_central_evaluation_function(model_template: ModelTemplate,
                                       evaluation_x_data: List[any],
                                       evaluation_y_data: List[any],
                                       optimizer: Optional[tf.keras.optimizers.Optimizer] = None
                                       ) -> EvalFunType:
    return __create_central_evaluation_function(
        model_template.get_model(),
        model_template.get_optimizer() if optimizer is None else optimizer,
        model_template.get_loss(),
        model_template.get_centralised_metrics(),
        evaluation_x_data,
        evaluation_y_data)


def create_central_evaluation_function_from_dataset(model_template: ModelTemplate,
                                                    dataset: Dataset,
                                                    client_dataset: ClientDataset,
                                                    optimizer: Optional[
                                                        tf.keras.optimizers.Optimizer] = None
                                                    ) -> EvalFunType:
    return create_central_evaluation_function(
        model_template,
        client_dataset.process_x(dataset.validation["x"]),
        client_dataset.process_y(dataset.validation["y"]),
        optimizer)


def __create_central_evaluation_function(model: tf.keras.Model,
                                         optimizer: Optional[tf.keras.optimizers.Optimizer],
                                         loss: tf.keras.losses.Loss,
                                         metrics: List[Union[str, tf.keras.metrics.Metric]],
                                         evaluation_x_data: List[any],
                                         evaluation_y_data: List[any]
                                         ) -> EvalFunType:

    return PickleableEvaluationFunction(model, optimizer, loss, metrics, evaluation_x_data,
                                        evaluation_y_data)
