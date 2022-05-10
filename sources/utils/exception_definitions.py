class OutsideOfContextError(Exception):
    pass


class ExperimentParameterListsHaveUnequalLengths(Exception):
    pass


class ConfigContainsUnknownPropertyError(Exception):
    pass


class DatasetNotFoundError(Exception):
    pass


class NoStrategyProviderError(Exception):
    pass


class LossNotDefinedError(Exception):
    pass


class LossDivergedError(Exception):
    pass


class AllClientLossesDivergedError(Exception):
    pass
