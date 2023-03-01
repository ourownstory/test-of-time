import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Type

import pandas as pd
from neuralprophet import NeuralProphet, TorchProphet

from tot.df_utils import _check_min_df_len, add_first_inputs_to_df, drop_first_inputs_from_df, prep_or_copy_df
from tot.models.models import Model
from tot.models.utils import _get_seasons

log = logging.getLogger("tot.model")


@dataclass
class NeuralProphetModel(Model):
    model_name: str = "NeuralProphet"
    model_class: Type = NeuralProphet

    def __post_init__(self):
        data_params = self.params["_data_params"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        if "seasonality_mode" in data_params and data_params["seasonality_mode"] is not None:
            self.params.update({"seasonality_mode": data_params["seasonality_mode"]})
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags
        self.season_length = None

    def fit(self, df: pd.DataFrame, freq: str):
        """Fits the model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns "ds" and "y" and optionally "ID"
        freq : str
            Frequency of the time series

        Returns
        -------
        None
        """
        _check_min_df_len(df=df, min_len=self.n_forecasts + self.n_lags)
        self.freq = freq
        _ = self.model.fit(df=df, freq=freq, progress="none", minimal=True)

    def predict(
        self,
        received_single_time_series,
        df: pd.DataFrame,
        df_historic: pd.DataFrame = None,
    ):
        """Runs the model to make predictions.

        Expects all data to be present in dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns "ds" and "y" and optionally "ID"
        received_single_time_series : bool
            Whether the df has only one time series
        df_historic : pd.DataFrame
            DataFrame containing column ``ds``, ``y``, and optionally ``ID`` with historic data

        Returns
        -------
        pd.DataFrame
            DataFrame with columns "ds", "y", "yhat1" and "ID"
        """
        if df_historic is not None:
            df = self.maybe_extend_df(df_train=df_historic, df_test=df)
        _check_min_df_len(df=df, min_len=self.n_forecasts + self.n_lags)
        fcst = self.model.predict(df=df)
        # add ID again since NP drops it
        (
            fcst,
            _,
            _,
            _,
        ) = prep_or_copy_df(fcst)

        if df_historic is not None:
            fcst = self.maybe_drop_added_values_from_df(fcst, df)
        return fcst

    def maybe_extend_df(self, df_train, df_test):
        """
        If model depends on historic values, extend beginning of df_test with last
        df_train values.
        """
        samples = self.n_lags
        df_test = add_first_inputs_to_df(samples=samples, df_train=df_train, df_test=df_test)
        return df_test

    def maybe_drop_added_values_from_df(self, predicted, df):
        """
        If model depends on historic values, drop first values of predicted and df_test.
        """
        samples = self.n_lags
        predicted = drop_first_inputs_from_df(samples=samples, predicted=predicted, df=df)

        return predicted


@dataclass
class TorchProphetModel(NeuralProphetModel):
    model_name: str = "TorchProphet"
    model_class: Type = TorchProphet

    def __post_init__(self):
        data_params = self.params["_data_params"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        if "seasonality_mode" in data_params and data_params["seasonality_mode"] is not None:
            self.params.update({"seasonality_mode": data_params["seasonality_mode"]})
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        # TorchProphet does not support uncertainty
        model_params.update({"interval_width": 0})
        # TorchProphet does not support n_forecasts>1 and n_lags>0
        if "n_forecasts" in model_params:
            assert model_params.n_forecasts == 1, "TorchProphet does not support n_forecasts >1."
        if "n_lags" in model_params:
            assert model_params.n_lags == 0, "TorchProphet does not support n_lags >0."

        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        # set fixed values for parent class methods
        self.n_forecasts = 1
        self.n_lags = 0

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
