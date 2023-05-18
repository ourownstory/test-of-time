import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Type

import pandas as pd
from neuralprophet import NeuralProphet, TorchProphet

from tot.df_utils import _check_min_df_len, add_first_inputs_to_df, drop_first_inputs_from_df, prep_or_copy_df
from tot.error_utils import raise_if
from tot.models.models import Model
from tot.models.utils import _get_seasons

log = logging.getLogger("tot.model")


@dataclass
class NeuralProphetModel(Model):
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

        # identifiy model_params
        if "lagged_regressors" in model_params.keys():
            model_params.pop("lagged_regressors")
        if "future_regressors" in model_params.keys():
            model_params.pop("future_regressors")
        self.model = self.model_class(**model_params)

        # map lagged regressors
        lagged_regressors = self.params.get("lagged_regressors", None)
        if isinstance(lagged_regressors, dict) is False and lagged_regressors is not None:
            lagged_regressors_dict = {}
            for lagged_regressor in lagged_regressors:
                lagged_regressors_dict.update({lagged_regressor: {}})
            lagged_regressors = lagged_regressors_dict
        if lagged_regressors is not None:
            for lagged_regressor in lagged_regressors.keys():
                self.model.add_lagged_regressor(
                    names=lagged_regressor, **lagged_regressors[lagged_regressor]
                ) if lagged_regressors[lagged_regressor] is not None else self.model.add_lagged_regressor(
                    names=lagged_regressor
                )

        # map future regressors
        future_regressors = self.params.get("future_regressors", None)
        if isinstance(future_regressors, dict) is False and future_regressors is not None:
            future_regressors_dict = {}
            for future_regressor in future_regressors:
                future_regressors_dict.update({future_regressor: {}})
            future_regressors = future_regressors_dict
        if future_regressors is not None:
            for future_regressor in future_regressors.keys():
                self.model.add_future_regressor(
                    name=future_regressor, **future_regressors[future_regressor]
                ) if future_regressors[future_regressor] is not None else self.model.add_future_regressor(
                    name=future_regressor
                )

        # map custom_seasonalities
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags
        self.season_length = None

    def fit(self, df: pd.DataFrame, freq: str, ids_weights: dict = None):
        """Fits the model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns "ds" and "y" and optionally "ID"
        freq : str
            Frequency of the time series
        ids_weights : str
            Weights per ID applied to the loss function.

        Returns
        -------
        None
        """
        _check_min_df_len(df=df, min_len=self.n_forecasts + self.n_lags)
        self.freq = freq
        _ = self.model.fit(df=df, freq=freq, progress="none", minimal=True, ids_weights=ids_weights)

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
        df_test = add_first_inputs_to_df(samples=self.n_lags, df_train=df_train, df_test=df_test)
        return df_test

    def maybe_drop_added_values_from_df(self, predicted, df):
        """
        If model depends on historic values, drop first values of predicted and df_test.
        """
        predicted = drop_first_inputs_from_df(samples=self.n_lags, predicted=predicted, df=df)

        return predicted


@dataclass
class TorchProphetModel(NeuralProphetModel):
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
        raise_if(
            "n_forecasts" in model_params and model_params["n_forecasts"] > 1,
            "TorchProphet does not support " "n_forecasts >1.",
        )
        raise_if("n_lags" in model_params and model_params["n_lags"] > 0, "TorchProphet does not support n_lags >0.")

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
