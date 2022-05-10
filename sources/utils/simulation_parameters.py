import logging
import typing


class SimulationParameters(typing.TypedDict):
    num_clients: int
    num_rounds: int


class EarlyStoppingSimulationParameters(SimulationParameters):
    target_accuracy: float
    num_rounds_above_target: int


class RayInitArgs(typing.TypedDict):
    ignore_reinit_error: bool
    include_dashboard: bool


class ClientResources(typing.TypedDict):
    num_cpus: int
    num_gpus: int


DEFAULT_RAY_INIT_ARGS: typing.Union[RayInitArgs, typing.Dict[str, bool]] = {
    "ignore_reinit_error": True,
    "include_dashboard": True,
    "address": None,
    "logging_level": logging.INFO,
}

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = "23000"
DEFAULT_SERVER_ADDRESS = DEFAULT_SERVER_HOST + ":" + DEFAULT_SERVER_PORT

DEFAULT_RUNS_PER_EXPERIMENT: int = 5
