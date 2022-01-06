import tensorflow as tf


class SparseMetricDecorator(tf.keras.metrics.Metric):
    def __init__(self, metric: tf.keras.metrics.Metric):
        self.metric = metric
        self.num_classes = metric.num_classes
        super().__init__(name=self.metric.name, dtype=self.metric.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, depth=self.num_classes, on_value=1, off_value=0)
        y_true = tf.reshape(y_true, shape=(-1, self.num_classes))
        return self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.metric.result()

    def get_config(self):
        return self.metric.get_config()

    def reset_state(self):
        return self.metric.reset_state()

    def reset_states(self):
        return self.metric.reset_states()
