from typing import Tuple, List

from flwr.common import Parameters, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from sources.flwr_strategies_decorators.base_strategy_decorator import BaseStrategyDecorator


class EnableFullEvaluationDecorator(BaseStrategyDecorator):
    """
    This function will enable full evaluation if the configure_evaluate function is based on flowers
    FedAvg Implementation in Flwr version 0.18.

    This must be the first Decorator used!!!!!
    """

    def configure_evaluate(self, rnd: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, EvaluateIns]]:
        config = self.strategy.configure_evaluate(rnd, parameters, client_manager)

        if config == [] or config is None:
            if self.is_first_decorator():
                config = self.default_evaluation_configuration_function(rnd, parameters, client_manager)

        return config

    def annotate_strategy_name(self, strategy_name: str) -> str:
        return "FullEval_" + strategy_name

    def default_evaluation_configuration_function(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Reenable Aggregation even if a central evaluation function is defined
        # if self.eval_fn is not None:
        #     return []

        # Parameters and config
        config = {}
        if self.strategy.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.strategy.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.strategy.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def is_first_decorator(self):
        return not hasattr(self.strategy, "strategy")