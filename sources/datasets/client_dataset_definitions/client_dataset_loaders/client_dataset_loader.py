from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import TypedDict

from numpy.typing import ArrayLike


class DatasetComponents(IntEnum):
    TRAIN = auto()
    TEST = auto()
    VALIDATION = auto()


class MinimalDataset(TypedDict):
    x: ArrayLike
    y: ArrayLike


class ClientDatasetLoader(ABC):

    @abstractmethod
    def load_dataset(
            self,
            client_identifier: str,
            dataset_component: DatasetComponents
    ) -> MinimalDataset:
        pass


