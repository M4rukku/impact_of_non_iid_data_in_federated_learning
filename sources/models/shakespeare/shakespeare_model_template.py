import tensorflow as tf
from typing import List, Union, Optional

from sources.global_data_properties import LEAF_CHARACTERS
from sources.metrics.default_metrics_tf import get_default_categorical_metrics_tf
from sources.models.keras_model_template import KerasModelTemplate

SHAKESPEARE_SEQ_LEN: int = 80
SHAKESPEARE_HIDDEN_SIZE: int = 256
SHAKESPEARE_EMBEDDING_DIM: int = 8


class ShakespeareKerasModelTemplate(KerasModelTemplate):

    def __init__(self, seed,
                 alphabet=LEAF_CHARACTERS,
                 seq_len=SHAKESPEARE_SEQ_LEN,
                 n_hidden=SHAKESPEARE_HIDDEN_SIZE,
                 embedding_dim=SHAKESPEARE_EMBEDDING_DIM,
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 ):
        self.seq_length = seq_len
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.alphabet = alphabet

        super(ShakespeareKerasModelTemplate, self).__init__(seed, loss, len(self.alphabet))

    def get_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.seq_length, dtype=tf.int32)

        embedding = tf.keras.layers.Embedding(len(self.alphabet), self.embedding_dim)(inputs)

        rnn_cells = [tf.keras.layers.LSTMCell(self.n_hidden) for _ in range(2)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        lstm_layer = tf.keras.layers.RNN(stacked_lstm)(embedding)

        dense = tf.keras.layers.Dense(units=len(self.alphabet))(lstm_layer)
        softmax = tf.keras.layers.Softmax()(dense)

        return tf.keras.Model(inputs=inputs, outputs=softmax)

    def get_centralised_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        return get_default_categorical_metrics_tf(self.num_classes)

    def get_optimizer(self, lr=1.0, model: Optional[tf.keras.models.Model] = None) \
            -> tf.keras.optimizers.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)
