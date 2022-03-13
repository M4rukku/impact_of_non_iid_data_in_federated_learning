import copy
import inspect
from typing import TypedDict, Optional, Union

from sources.experiments.experiment_metadata import ExperimentMetadata


class FixedExperimentMetadata(TypedDict):
    num_clients: Union[int, None]
    num_rounds: int
    clients_per_round: int
    batch_size: int
    local_epochs: int
    val_steps: int


class FixedExperimentMetadataEarlyStopping(FixedExperimentMetadata):
    target_accuracy: float
    num_rounds_above_target: int


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs


class ExperimentMetadataProvider:
    def __init__(self, fixed_metadata: FixedExperimentMetadata):
        self.fixed_metadata = fixed_metadata

    def __call__(self,
                 num_clients: int = None,
                 num_rounds: int = None,
                 clients_per_round: int = None,
                 batch_size: int = None,
                 local_epochs: int = None,
                 val_steps: int = None,
                 target_accuracy: Optional[float] = None,
                 num_rounds_above_target: int = None,
                 custom_suffix=None,
                 **kwargs
                 ) -> ExperimentMetadata:
        all_kwargs = get_kwargs()
        cpy = copy.deepcopy(self.fixed_metadata)
        new_args_wo_none = {key: val for key, val in all_kwargs.items() if val is not None and key not in kwargs}
        cpy.update(new_args_wo_none)
        return ExperimentMetadata(**cpy)
