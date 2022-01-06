import copy
import inspect
from enum import IntEnum, auto
from typing import TypedDict

from sources.experiments.experiment_metadata import ExperimentMetadata


class ExperimentScale(IntEnum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()


class FixedExperimentMetadata(TypedDict):
    num_rounds: int
    clients_per_round: int
    batch_size: int
    local_epochs: int
    val_steps: int


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs


class _ExperimentMetadataProvider:
    def __init__(self, fixed_metadata: FixedExperimentMetadata):
        self.fixed_metadata = fixed_metadata

    def __call__(self,
                 num_clients: int,
                 num_rounds: int = None,
                 clients_per_round: int = None,
                 batch_size: int = None,
                 local_epochs: int = None,
                 val_steps: int = None) -> ExperimentMetadata:
        kwargs = get_kwargs()
        cpy = copy.deepcopy(self.fixed_metadata)
        new_args_wo_none = {key: val for key, val in kwargs.items() if val is not None}
        cpy.update(new_args_wo_none)
        return ExperimentMetadata(**cpy)