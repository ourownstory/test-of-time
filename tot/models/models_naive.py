import logging
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from tot.df_utils import _check_min_df_len, add_first_inputs_to_df, drop_first_inputs_from_df
from tot.models.models import Model
from tot.models.utils import _convert_seasonality_to_season_length, _get_seasons, _predict_seasonal_naive

log = logging.getLogger("tot.model")


@dataclass
class SeasonalNaiveModel(Model):
    """
    A `SeasonalNaiveModel` is a naive model that forecasts future values of a target series based on past observations
    of the target series of the specified period, i.e season.

    Parameters
    ----------
        season_length : int
            seasonal period in number of time steps
        n_forecasts : int
            number of steps ahead of prediction time step to forecast
    """

    model_name: str = "SeasonalNaive"

    def __post_init__(self):
        # no installation checks required

        # re-assign _data_params
        data_params = self.params["_data_params"]
        self.freq = data_params["freq"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        if "seasonality_mode" in data_params and data_params["seasonality_mode"] is not None:
            self.params.update({"seasonality_mode": data_params["seasonality_mode"]})

        # Verify expected model_params: season_length > 1, n_forecasts >=1
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.n_forecasts = model_params["n_forecasts"]
        if self.n_forecasts < 1:
            raise ValueError("Model parameter n_forecasts must be >=1.")

        self.season_length = None
        # always select seasonality provided by dataset first
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            self.season_length = _convert_seasonality_to_season_length(
                data_params["freq"],
                daily,
                weekly,
                yearly,
                custom_seasonalities,
            )
        elif "season_length" in model_params:
            self.season_length = model_params["season_length"]  # for seasonal naive season_length is input parameter
        if self.season_length is None:
            raise ValueError(
                "Dataset does not provide a seasonality. Assign a seasonality to each of the datasets "
                "OR input desired season_length as model parameter to be used for all datasets "
                "without specified seasonality."
            )
        if self.season_length <= 1:
            raise ValueError(
                "season_length must be >1 for SeasonalNaiveModel. For season_length=1 select NaiveModel " "instead."
            )

    def fit(self, df: pd.DataFrame, freq: str):
        pass

    def predict(self, df: pd.DataFrame, received_single_time_series, df_historic: pd.DataFrame = None):
        """Runs the model to make predictions.
        Expects all data to be present in dataframe.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
            df_historic : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with historic data
            received_single_time_series : bool
                whether it is a single time series

        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``, optionally [``ID``], and [``yhat<i>``] where yhat<i> refers to the
                i-step-ahead prediction for this row's datetime, e.g. yhat3 is the prediction for this datetime,
                predicted 3 steps ago, "3 steps old".

                Note
                ----
                 *  raw data is not supported
        """
        if df_historic is not None:
            df = self.maybe_extend_df(df_train=df_historic, df_test=df)
        _check_min_df_len(df=df, min_len=self.n_forecasts + self.season_length)
        fcst = _predict_seasonal_naive(df=df, season_length=self.season_length, n_forecasts=self.n_forecasts)

        if df_historic is not None:
            fcst = self.maybe_drop_added_values_from_df(fcst, df)
        return fcst

    def maybe_extend_df(self, df_train, df_test):
        """
        If model depends on historic values, extend beginning of df_test with last
        df_train values.
        """
        samples = self.season_length
        df_test = add_first_inputs_to_df(samples=samples, df_train=df_train, df_test=df_test)

        return df_test

    def maybe_drop_added_values_from_df(self, predicted, df):
        """
        If model depends on historic values, drop first values of predicted and df_test.
        """
        samples = self.season_length
        predicted = drop_first_inputs_from_df(samples=samples, predicted=predicted, df=df)
        return predicted


@dataclass()
class NaiveModel(SeasonalNaiveModel):
    """
    A `NaiveModel` is a naive model that forecasts future values of a target series as the value of the
    last observation of the target series. The NaiveModel is SeasonalNaiveModel with K=1.

    Parameters
    ----------
        n_forecasts : int
            number of steps ahead of prediction time step to forecast
    Raises
    -------
        ValueError
            If Model parameter n_forecasts is less than 1.
    """

    model_name: str = "NaiveModel"

    def __post_init__(self):
        # no installation checks required

        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.n_forecasts = model_params["n_forecasts"]
        if self.n_forecasts < 1:
            raise ValueError("Model parameter n_forecasts must be >=1.")
        self.season_length = 1  # season_length=1 for NaiveModel
