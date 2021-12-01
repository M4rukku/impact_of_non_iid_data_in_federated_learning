import typing


class FederatedEvaluationParameters(typing.TypedDict):
    batch_size: int
    val_steps: int


class FittingParameters(typing.TypedDict):
    batch_size: int
    local_epochs: int
