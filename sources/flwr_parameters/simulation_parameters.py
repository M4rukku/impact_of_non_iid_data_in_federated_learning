import typing


class SimulationParameters(typing.TypedDict):
    num_clients: int
    num_rounds: int


DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = "11111"
DEFAULT_SERVER_ADDRESS = DEFAULT_SERVER_HOST + ":" + DEFAULT_SERVER_PORT