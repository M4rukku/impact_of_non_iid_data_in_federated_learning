import copy
import logging
from typing import Optional, Tuple, Dict, Union, List, Callable
import numpy
import tensorflow as tf

from sources.utils.dataset import Dataset
from sources.datasets.client_dataset_definitions.client_dataset_processors.client_dataset_processor import \
    ClientDatasetProcessor
from sources.models.keras_model_template import KerasModelTemplate

EvalFunType = Callable[[List[numpy.ndarray]],
                       Optional[Tuple[float, Dict[str, Union[bool, bytes, float, int, str]]]]]


class PickleableKerasCentralEvaluationFunction:
    def __init__(self, model_template, optimizer, loss, metrics, evaluation_x_data,
                 evaluation_y_data):
        # make_keras_pickleable() Obsolete for Keras 2.3
        self.model_template = model_template
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.evaluation_x_data = evaluation_x_data
        self.evaluation_y_data = evaluation_y_data

    def __call__(self, results: List[numpy.ndarray]) -> \
            Optional[Tuple[float, Dict[str, Union[bool, bytes, float, int, str]]]]:
        # set_global_determinism()
        logging.warning("Starting Centralised Evaluation")

        model = self.model_template.get_model()
        optimizer_ = copy.deepcopy(self.optimizer)
        loss_ = copy.deepcopy(self.loss)
        metrics_ = copy.deepcopy(self.metrics)

        model.compile(optimizer=optimizer_,
                      loss=loss_,
                      metrics=metrics_)

        model.set_weights(results)
        results = model.evaluate(x=self.evaluation_x_data, y=self.evaluation_y_data,
                                 return_dict=True)
        logging.warning(f"Finished Centralised Evaluation with result {results.items()}")
        return results["loss"], results


def create_central_evaluation_function_keras(model_template: KerasModelTemplate,
                                             evaluation_x_data: List[any],
                                             evaluation_y_data: List[any],
                                             optimizer: Optional[
                                                 tf.keras.optimizers.Optimizer] = None
                                             ) -> EvalFunType:
    return __create_central_evaluation_function_keras(
        model_template,
        model_template.get_optimizer() if optimizer is None else optimizer,
        model_template.get_loss(),
        model_template.get_centralised_metrics(),
        evaluation_x_data,
        evaluation_y_data)


def create_central_evaluation_function_from_dataset_processor_keras(
        model_template: KerasModelTemplate,
        dataset: Dataset,
        client_dataset_processor: ClientDatasetProcessor,
        optimizer: Optional[
            tf.keras.optimizers.Optimizer] = None
) -> EvalFunType:
    return create_central_evaluation_function_keras(
        model_template,
        client_dataset_processor.process_x(dataset.validation["x"]),
        client_dataset_processor.process_y(dataset.validation["y"]),
        optimizer)


def __create_central_evaluation_function_keras(model_template: KerasModelTemplate,
                                               optimizer: Optional[tf.keras.optimizers.Optimizer],
                                               loss: tf.keras.losses.Loss,
                                               metrics: List[Union[str, tf.keras.metrics.Metric]],
                                               evaluation_x_data: List[any],
                                               evaluation_y_data: List[any]
                                               ) -> EvalFunType:
    model_template_ = copy.deepcopy(model_template)
    optimizer_ = copy.deepcopy(optimizer)
    loss_ = copy.deepcopy(loss)
    metrics_ = copy.deepcopy(metrics)
    return PickleableKerasCentralEvaluationFunction(model_template_,
                                                    optimizer_,
                                                    loss_,
                                                    metrics_,
                                                    evaluation_x_data,
                                                    evaluation_y_data)
