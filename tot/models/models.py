import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger("tot.model")


@dataclass
class Model(ABC):
    """
    example use:
    >>> models = []
    >>> for params in [{"n_changepoints": 5}, {"n_changepoints": 50},]:
    >>>     models.append(Model(
    >>>         params=params
    >>>         model_name="NeuralProphet",
    >>>         model_class=NeuralProphet,
    >>>     ))
    """

    params: dict
    model_name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, freq: str):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, received_single_time_series, df_historic: pd.DataFrame = None):
        pass

    def maybe_extend_df(self, df_train, df_test):
        """
        If model depends on historic values, extend beginning of df_test with last
        df_train values.
        """
        return df_test

    def maybe_drop_added_values_from_df(self, predicted, df):
        """
        If model depends on historic values, drop first values of predicted and df_test.
        """
        return predicted
