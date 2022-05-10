from datetime import datetime
from logging import log, INFO, DEBUG, WARNING
from typing import Optional, Tuple, Dict, Union, List

from flwr.common import Scalar, Parameters, weights_to_parameters, FitIns, FitRes, EvaluateIns, \
    EvaluateRes, Weights
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import EvaluateResultsAndFailures, DEPRECATION_WARNING_EVALUATE_ROUND, \
    FitResultsAndFailures, DEPRECATION_WARNING_FIT_ROUND

from sources.utils.exception_definitions import AllClientLossesDivergedError, LossDivergedError


class SerialExecutionServer(Server):

    def evaluate_round(
            self, rnd: int
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, f"evaluate_round {rnd}: no clients selected, cancel")
            return None
        log(
            INFO,
            f"evaluate_round {rnd}: Time {datetime.now().time()}: strategy sampled %s clients ("
            f"out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(client_instructions)
        log(
            INFO,
            "evaluate_round {rnd}: Time {datetime.now().time()}: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Union[
            Tuple[Optional[float], Dict[str, Scalar]],
            Optional[float],  # Deprecated
        ] = self.strategy.aggregate_evaluate(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = None
        elif isinstance(aggregated_result, float):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = aggregated_result
        else:
            loss_aggregated, metrics_aggregated = aggregated_result

        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
            self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, f"fit_round {rnd}: Time {datetime.now().time()}: no clients selected, cancel")
            return None
        log(
            INFO,
            f"fit_round {rnd}: Time {datetime.now().time()}: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(client_instructions)
        log(
            INFO,
            f"fit_round {rnd}: Time {datetime.now().time()}: received %s results and %s failures",
            len(results),
            len(failures),
        )

        if len(results) == 0:
            all_clients_diverged = True
            for failure in failures:
                failure: BaseException = failure
                if not isinstance(failure, LossDivergedError):
                    all_clients_diverged = False
                    break
            if all_clients_diverged:
                raise AllClientLossesDivergedError("AllClientLossesDivergedError - Stopping "
                                                   "Execution of Run")

        # Aggregate training results
        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[Weights],  # Deprecated
        ] = self.strategy.aggregate_fit(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = None
        elif isinstance(aggregated_result, list):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = weights_to_parameters(aggregated_result)
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)


def fit_clients(
        client_instructions: List[Tuple[ClientProxy, FitIns]]
) -> FitResultsAndFailures:
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []

    for c, ins in client_instructions:
        try:
            results.append(fit_client(c, ins))
        except BaseException as e:
            failures.append(e)

    return results, failures


def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res


def evaluate_clients(
        client_instructions: List[Tuple[ClientProxy, EvaluateIns]]
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []

    for c, ins in client_instructions:
        try:
            results.append(evaluate_client(c, ins))
        except BaseException as e:
            failures.append(e)

    return results, failures


def evaluate_client(
        client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
