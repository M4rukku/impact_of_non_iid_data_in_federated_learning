import typing


class SimulationParameters(typing.TypedDict):
    num_clients: int
    num_rounds: int


class RayInitArgs(typing.TypedDict):
    ignore_reinit_error: bool
    include_dashboard: bool


class ClientResources(typing.TypedDict):
    num_cpus: int
    num_gpus: int


DEFAULT_RAY_INIT_ARGS: RayInitArgs = {
    "ignore_reinit_error": True,
    "include_dashboard": True
}

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = "23000"
DEFAULT_SERVER_ADDRESS = DEFAULT_SERVER_HOST + ":" + DEFAULT_SERVER_PORT
