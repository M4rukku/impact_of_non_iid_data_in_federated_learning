import logging

from flwr.server import SimpleClientManager
from flwr.server.app import _fl

from sources.simulators.serial_execution_simulator.serial_execution_client_proxy import \
    SerialExecutionClientProxy
from sources.simulators.serial_execution_simulator.serial_execution_server import \
    SerialExecutionServer
from sources.utils.exception_definitions import AllClientLossesDivergedError


def start_serial_simulation(client_fn, clients_ids, num_rounds, strategy):
    # Initialize server and server config
    config = {"num_rounds": num_rounds}
    initialized_server, initialized_config = \
        (SerialExecutionServer(SimpleClientManager(), strategy), config)

    logging.log(
        logging.INFO,
        "Starting Flower simulation running: %s",
        initialized_config,
    )

    # Register one RayClientProxy object for each client with the ClientManager
    for cid in clients_ids:
        client_proxy = SerialExecutionClientProxy(
            client_fn=client_fn,
            cid=cid
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    try:
        _fl(
            server=initialized_server,
            config=initialized_config,
            force_final_distributed_eval=False,
        )
    except AllClientLossesDivergedError as e:
        logging.error(f"Error while running current Simulation - All Clients Diverged during "
                      f"Fitting - Please check your hyperparameter combination. It might be "
                      f"unsuitable for the task at hand.")
        logging.error(str(e))
