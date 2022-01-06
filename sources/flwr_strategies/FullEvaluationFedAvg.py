from typing import Tuple, List, Callable

from flwr.common import Parameters, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from sources.experiments.experiment_metadata import ExperimentMetadata


class FullEvaluationFedAvg(FedAvg):

    def configure_evaluate(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Reenable Aggregation even if a central evaluation function is defined
        # if self.eval_fn is not None:
        #     return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


def full_eval_fed_avg_strategy_provider(
        eval_fn: Callable, experiment_metadata: ExperimentMetadata
) -> FullEvaluationFedAvg:
    if experiment_metadata.clients_per_round >= 1:
        fraction_fit = float(experiment_metadata.clients_per_round) / \
                       float(experiment_metadata.num_clients)
    else:
        fraction_fit = experiment_metadata.clients_per_round

    strategy = FullEvaluationFedAvg(eval_fn=eval_fn,
                                    fraction_fit=fraction_fit,
                                    fraction_eval=fraction_fit)

    return strategy