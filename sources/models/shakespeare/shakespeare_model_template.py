import tensorflow as tf

from global_data_properties import LEAF_CHARACTERS
from sources.models.model_template import ModelTemplate

SHAKESPEARE_SEQ_LEN: int = 80
SHAKESPEARE_HIDDEN_SIZE: int = 256
SHAKESPEARE_EMBEDDING_DIM: int = 8

class ShakespeareModelTemplate(ModelTemplate):

    def __init__(self, seed,
                 alphabet=LEAF_CHARACTERS,
                 seq_len=SHAKESPEARE_SEQ_LEN,
                 n_hidden=SHAKESPEARE_HIDDEN_SIZE,
                 embedding_dim=SHAKESPEARE_EMBEDDING_DIM,
                 optimizer=tf.keras.optimizers.SGD(),
                 loss=tf.keras.losses.CategoricalCrossentropy()
                 ):
        self.seq_length = seq_len
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.alphabet = alphabet

        super(ShakespeareModelTemplate, self).__init__(seed, optimizer, loss)

    def get_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.seq_length, dtype=tf.int32)

        embedding = tf.keras.layers.Embedding(len(self.alphabet), self.embedding_dim)(inputs)

        rnn_cells = [tf.keras.layers.LSTMCell(self.n_hidden) for _ in range(2)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        lstm_layer = tf.keras.layers.RNN(stacked_lstm)(embedding)

        dense = tf.keras.layers.Dense(units=len(self.alphabet))(lstm_layer)
        softmax = tf.keras.layers.Softmax()(dense)

        return tf.keras.Model(inputs=inputs, outputs=softmax)
