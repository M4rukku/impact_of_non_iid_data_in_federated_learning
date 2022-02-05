from typing import List
import numpy as np
import tensorflow as tf

from sources.datasets.client_dataset_definitions.client_dataset_processors.client_dataset_processor import ClientDatasetProcessor
from sources.global_data_properties import LEAF_CHARACTERS


class ShakespeareClientDatasetProcessor(ClientDatasetProcessor):
    def __init__(self, alphabet: str = LEAF_CHARACTERS):
        self.alphabet = alphabet
        self.character_integer_mapping = {c: i for i, c in enumerate(self.alphabet)}

    def _word_to_indices(self, word: str) -> List[int]:
        """Converts a sequence of characters into position indices in the
        reference string `self.characters`.

        Args:
            word (str): Sequence of characters to be converted.

        Returns:
            List[int]: List with positions.
        """
        indices: List[int] = [self.character_integer_mapping[c] for c in word]
        return indices

    def _letter_to_vec(self, letter):
        index = self.character_integer_mapping[letter]
        return tf.keras.utils.to_categorical(index, len(self.alphabet))

    def process_x(self, raw_x_batch):
        x_batch = [self._word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = np.array([self._letter_to_vec(c) for c in raw_y_batch])
        return y_batch
