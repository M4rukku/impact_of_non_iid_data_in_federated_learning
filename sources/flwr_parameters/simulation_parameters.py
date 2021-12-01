import typing


class SimulationParameters(typing.TypedDict):
    num_clients: int
    num_rounds: int


DEFAULT_SERVER_HOST = "[::]"
DEFAULT_SERVER_PORT = "9898"
DEFAULT_SERVER_ADDRESS = DEFAULT_SERVER_HOST + ":" + DEFAULT_SERVER_PORT