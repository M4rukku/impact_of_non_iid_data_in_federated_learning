from typing import TypedDict
import dataclasses
import numpy.typing as npt


class XYDict(TypedDict):
    x: npt.ArrayLike
    y: npt.ArrayLike


@dataclasses.dataclass
class Dataset:
    train: XYDict
    test: XYDict
    validation: XYDict

    def __iter__(self):
        yield self.train
        yield self.test
        yield self.validation

    def to_tuple(self):
        return self.train, self.test, self.validation
