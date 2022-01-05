import tensorflow_addons as tfa
from sources.metrics.sparse_metric_decorator import SparseMetricDecorator

DEFAULT_METRICS = ["accuracy"]


def get_default_sparse_categorical_metrics(num_classes: int):
    return ["accuracy",
            SparseMetricDecorator(tfa.metrics.MatthewsCorrelationCoefficient(num_classes)),
            SparseMetricDecorator(
                tfa.metrics.F1Score(num_classes, name="macro_f1_score", average='macro')
            ),
            SparseMetricDecorator(
                tfa.metrics.F1Score(num_classes, name="micro_f1_score", average='micro')
            )
            ]


def get_default_categorical_metrics(num_classes: int):
    return ["accuracy",
            tfa.metrics.MatthewsCorrelationCoefficient(num_classes),
            tfa.metrics.F1Score(num_classes, name="macro_f1_score", average='macro'),
            tfa.metrics.F1Score(num_classes, name="micro_f1_score", average='micro')
            ]
