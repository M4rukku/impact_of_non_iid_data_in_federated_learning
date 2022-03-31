import logging
from abc import ABC


class BaseModelTemplate(ABC):
    # Switch to set_hyperparameters in the future!

    def set_optimizer(self, param):
        raise NotImplementedError("If you wish to vary the optimiser using simulate_experiment, "
                                  "you have to implement set_optimizer, get_optimizer_config")

    def get_optimizer(self):
        return None

    def get_optimizer_config(self) -> str:
        logging.warning("If you wish to vary the optimiser using simulate_experiment, "
                        "you have to implement set_optimizer, get_optimizer_config")
        return ""
