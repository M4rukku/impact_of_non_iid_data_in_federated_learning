from typing import Union
import tensorflow as tf

from sources.models.model import Model

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


class ShakespeareClientModel(Model):

    def __init__(self, seed,
                 alphabet=LEAF_CHARACTERS,
                 seq_len=80,
                 n_hidden=256,
                 embedding_dim=8
                 ):
        self.seq_length = seq_len
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.alphabet = alphabet

        super(ShakespeareClientModel, self).__init__(seed)

    def get_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.seq_length, dtype=tf.int32)

        embedding = tf.keras.layers.Embedding(len(self.alphabet), self.embedding_dim)(inputs)

        rnn_cells = [tf.keras.layers.LSTMCell(self.n_hidden) for _ in range(2)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        lstm_layer = tf.keras.layers.RNN(stacked_lstm)(embedding)

        dense = tf.keras.layers.Dense(units=len(self.alphabet))(lstm_layer)
        softmax = tf.keras.layers.Softmax()(dense)

        return tf.keras.Model(inputs=inputs, outputs=softmax)

    def get_default_optimizer(self, lr: Union[None, float] = None) -> tf.keras.optimizers.Optimizer:
        if lr is not None:
            return tf.keras.optimizers.SGD(lr=lr)
        else:
            return tf.keras.optimizers.SGD()

    def get_default_loss_function(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.CategoricalCrossentropy()
