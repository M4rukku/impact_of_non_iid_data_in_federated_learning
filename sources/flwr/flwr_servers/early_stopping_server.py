# Strongly Modified from Adap

# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging

from flwr.server import Server, History
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.common.logger import log

import timeit
from logging import INFO
from typing import Optional

DEFAULT_NUM_ROUNDS_ABOVE_TARGET = 5


class EarlyStoppingServer(Server):

    def __init__(self,
                 client_manager: ClientManager,
                 strategy: Optional[Strategy],
                 target_accuracy: Optional[float] = None,
                 num_rounds_above_target: int = DEFAULT_NUM_ROUNDS_ABOVE_TARGET
                 ):
        super().__init__(client_manager, strategy)

        self.target_accuracy = target_accuracy
        self.num_rounds_above_target = num_rounds_above_target
        self.early_stopping_enabled = self.target_accuracy is not None

    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(rnd=current_round)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(rnd=current_round)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                    history.add_metrics_distributed(
                        rnd=current_round, metrics=evaluate_metrics_fed
                    )

            # Early Stopping on achieveing Target Accuracy for a small number of rounds
            if self.early_stopping_enabled:
                if res_fed is not None:
                    distributed_history_metrics = history.metrics_distributed

                    if "acc" in distributed_history_metrics:
                        dist_accuracy_metric_list = distributed_history_metrics["acc"]
                    elif "accuracy" in distributed_history_metrics:
                        dist_accuracy_metric_list = distributed_history_metrics["accuracy"]
                    else:
                        dist_accuracy_metric_list = None
                        log(logging.WARNING, f"FL-Target Accuracy Check - Round {current_round} : A target accuracy of "
                                             f"{self.target_accuracy} has been set, but the evaluation "
                                             f"dictionary does not contain accuracy (keyed by acc or accuracy)")

                    if dist_accuracy_metric_list is not None:
                        if len(dist_accuracy_metric_list) > self.num_rounds_above_target:
                            last_epoch_results = dist_accuracy_metric_list[-self.num_rounds_above_target:]
                            last_epoch_accuracies = list(map(lambda p: p[1], last_epoch_results))
                            if all(map(lambda v: v >= self.target_accuracy, last_epoch_accuracies)):
                                log(logging.WARNING,
                                    f"FL - Round {current_round} : A target accuracy of "
                                    f"{self.target_accuracy} has been reached for {self.num_rounds_above_target} rounds"
                                    f" -- Stopping the Simulation early.")
                                break
                else:
                    log(logging.WARNING, f"FL-Target Accuracy Check - Round {current_round} : A target accuracy of "
                                         f"{self.target_accuracy} has been set, but the result dictionary "
                                         f"is invalid (resfed is None)")

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
