import tensorflow_addons as tfa
from sources.metrics.sparse_metric_decorator_tf import SparseMetricDecoratorTensorflow

DEFAULT_METRICS = ["accuracy"]


def get_default_sparse_categorical_metrics_tf(num_classes: int):
    return ["accuracy",
            SparseMetricDecoratorTensorflow(tfa.metrics.MatthewsCorrelationCoefficient(num_classes)),
            SparseMetricDecoratorTensorflow(
                tfa.metrics.F1Score(num_classes, name="macro_f1_score", average='macro')
            ),
            SparseMetricDecoratorTensorflow(
                tfa.metrics.F1Score(num_classes, name="micro_f1_score", average='micro')
            )
            ]


def get_default_categorical_metrics_tf(num_classes: int):
    return ["accuracy",
            tfa.metrics.MatthewsCorrelationCoefficient(num_classes),
            tfa.metrics.F1Score(num_classes, name="macro_f1_score", average='macro'),
            tfa.metrics.F1Score(num_classes, name="micro_f1_score", average='micro')
            ]
