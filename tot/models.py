import logging
import numpy as np
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import pandas as pd
from neuralprophet import NeuralProphet, df_utils
from tot.utils import convert_to_datetime, _get_seasons

try:
    from prophet import Prophet

    _prophet_installed = True
except ImportError:
    Prophet = None
    _prophet_installed = False

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
    model_class: Type

    @abstractmethod
    def fit(self, df: pd.DataFrame, freq: str):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        pass

    def maybe_add_first_inputs_to_df(self, df_train, df_test):
        """
        if Model with lags: adds n_lags values to start of df_test.
        else (time-features only): returns unchanged df_test
        """
        return df_test.reset_index(drop=True)

    def maybe_drop_first_forecasts(self, predicted, df):
        """
        if Model with lags: removes first n_lags values from predicted and df_test
        else (time-features only): returns unchanged df_test
        """
        return predicted.reset_index(drop=True), df.reset_index(drop=True)

    def maybe_drop_added_dates(self, predicted, df):
        """if Model imputed any dates: removes any dates in predicted which are not in df_test."""
        return predicted.reset_index(drop=True), df.reset_index(drop=True)


@dataclass
class ProphetModel(Model):
    model_name: str = "Prophet"
    model_class: Type = Prophet

    def __post_init__(self):
        if not _prophet_installed:
            raise RuntimeError("Requires prophet to be installed")
        data_params = self.params["_data_params"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(name="{}_daily".format(str(seasonality)), period=seasonality)
        self.n_forecasts = 1
        self.n_lags = 0

    def fit(self, df: pd.DataFrame, freq: str):
        if "ID" in df.columns and len(df["ID"].unique()) > 1:
            raise NotImplementedError("Prophet does not work with many ts df")
        self.freq = freq
        self.model = self.model.fit(df=df)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        fcst_df = pd.DataFrame({"time": fcst.ds, "y": df.y, "yhat1": fcst.yhat})
        return fcst_df


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
                self.model.add_seasonality(name="{}_daily".format(str(seasonality)), period=seasonality)
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        _ = self.model.fit(df=df, freq=freq, progress="none", minimal=True)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        fcst, received_ID_col, received_single_time_series, received_dict, _ = df_utils.prep_or_copy_df(fcst)
        fcst_df = pd.DataFrame()
        for df_name, fcst_i in fcst.groupby("ID"):
            y_cols = ["y"] + [col for col in fcst_i.columns if "yhat" in col]
            fcst_aux = pd.DataFrame({"time": fcst_i.ds})
            for y_col in y_cols:
                fcst_aux[y_col] = fcst_i[y_col]
            fcst_aux["ID"] = df_name
            fcst_df = pd.concat((fcst_df, fcst_aux), ignore_index=True)
        fcst_df = df_utils.return_df_in_original_format(
            fcst_df, received_ID_col, received_single_time_series, received_dict
        )
        return fcst_df

    def maybe_add_first_inputs_to_df(self, df_train, df_test):
        """Adds last n_lags values from df_train to start of df_test."""
        if self.model.n_lags > 0:
            df_train, _, _, _, _ = df_utils.prep_or_copy_df(df_train)
            (
                df_test,
                received_ID_col_test,
                received_single_time_series_test,
                received_dict_test,
                _,
            ) = df_utils.prep_or_copy_df(df_test)
            df_test_new = pd.DataFrame()
            for df_name, df_test_i in df_test.groupby("ID"):
                df_train_i = df_train[df_train["ID"] == df_name].copy(deep=True)
                df_test_i = pd.concat([df_train_i.tail(self.model.n_lags), df_test_i], ignore_index=True)
                df_test_new = pd.concat((df_test_new, df_test_i), ignore_index=True)
            df_test = df_utils.return_df_in_original_format(
                df_test_new, received_ID_col_test, received_single_time_series_test, received_dict_test
            )
        return df_test

    def maybe_drop_first_forecasts(self, predicted, df):
        """
        if Model with lags: removes first n_lags values from predicted and df
        else (time-features only): returns unchanged df
        """
        if self.model.n_lags > 0:
            (
                predicted,
                received_ID_col_pred,
                received_single_time_series_pred,
                received_dict_test_pred,
                _,
            ) = df_utils.prep_or_copy_df(predicted)
            df, received_ID_col_df, received_single_time_series_df, received_dict_test_df, _ = df_utils.prep_or_copy_df(
                df
            )
            predicted_new = pd.DataFrame()
            df_new = pd.DataFrame()
            for df_name, df_i in df.groupby("ID"):
                predicted_i = predicted[predicted["ID"] == df_name].copy(deep=True)
                predicted_i = predicted_i[self.model.n_lags :]
                df_i = df_i[self.model.n_lags :]
                df_new = pd.concat((df_new, df_i), ignore_index=True)
                predicted_new = pd.concat((predicted_new, predicted_i), ignore_index=True)
            df = df_utils.return_df_in_original_format(
                df_new, received_ID_col_df, received_single_time_series_df, received_dict_test_df
            )
            predicted = df_utils.return_df_in_original_format(
                predicted_new, received_ID_col_pred, received_single_time_series_pred, received_dict_test_pred
            )
        return predicted, df

    def maybe_drop_added_dates(self, predicted, df):
        """if Model imputed any dates: removes any dates in predicted which are not in df_test."""
        (
            predicted,
            received_ID_col_pred,
            received_single_time_series_pred,
            received_dict_test_pred,
            _,
        ) = df_utils.prep_or_copy_df(predicted)
        df, received_ID_col_df, received_single_time_series_df, received_dict_test_df, _ = df_utils.prep_or_copy_df(df)
        predicted_new = pd.DataFrame()
        df_new = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            predicted_i = predicted[predicted["ID"] == df_name].copy(deep=True)
            df_i["ds"] = convert_to_datetime(df_i["ds"])
            df_i.set_index("ds", inplace=True)
            predicted_i.set_index("time", inplace=True)
            predicted_i = predicted_i.loc[df_i.index]
            predicted_i = predicted_i.reset_index()
            df_i = df_i.reset_index()
            df_new = pd.concat((df_new, df_i), ignore_index=True)
            predicted_new = pd.concat((predicted_new, predicted_i), ignore_index=True)
        df = df_utils.return_df_in_original_format(
            df_new, received_ID_col_df, received_single_time_series_df, received_dict_test_df
        )
        predicted = df_utils.return_df_in_original_format(
            predicted_new, received_ID_col_pred, received_single_time_series_pred, received_dict_test_pred
        )
        return predicted, df


@dataclass
class SeasonalNaiveModel(Model):
    """
    Seasonal Naive model
    TODO: add capabilities
    """

    model_name: str = "SeasonalNaiveModel"
    model_class: Type = None  # make sense?

    def __post_init__(self):
        # TODO: any installation checks?
        data_params = self.params["_data_params"]
        custom_seasonalities = None  # TODO: relevant for seasonal naive

        # TODO: check what is self.model
        self.n_forecasts = self.params["n_forecasts"]
        self.n_lags = 0  # seasonal naive model does not support autoregression
        self.K = self.params["K"]  # for seasonal naive K is input parameter

    def fit(self, df: pd.DataFrame, freq: str):
        """Fits the seasonal naive model.
        Seasonal Naive model does not need to be explicitly fitted. However, we store fitting-related information.
        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and ``ID`` with all data
            freq : str
                frequency of the input data
        """
        if "ID" in df.columns and len(df["ID"].unique()) > 1:
            raise NotImplementedError("NaiveModel does not work with many ts df")
        if df.shape[0] <= self.K:
            raise ValueError(f"The time series requires at least K={self.K} points")

        self.freq = freq
        # min. number of prebious observations to base first naive prediction on
        self.min_observations = max(3, self.K)
        # TODO: edit length of df_test based on K
        # TODO: add auto-seasonality

    def predict(self, df: pd.DataFrame):
        """Runs the model to make predictions.
        Expects all data needed to be present in dataframe.
        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and ``ID`` with data
        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``, and [``yhat<i>``] where yhat<i> refers to the i-step-ahead prediction for this
                row's datetime, e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
                Note
                ----
                 *  raw data is not supported
        """
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1

        # TODO: add prep_or_copy_df(df)
        fcst_df = pd.DataFrame
        for df_name, df_i in df.groupby("ID"):
            dates, predicted = self._predict_raw(df_i, df_name)
            # TODO: add raw prediction option based on dates
            fcst_df = self._reshape_raw_predictions_to_forecst_df(df_i, predicted)
            # TODO: add method to return df in original format? i.e. drop ID column
        return fcst_df

    def _predict_raw(self, df, df_name):
        """Computes forecast-origin-wise seasonal naive predictions.
        Predictions are returned in raw vector format. Predictions are given on a forecast origin basis,
        not on a target basis.
        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            df_name : str
                name of the data params from which the current dataframe refers to (only in case of local_normalization)
        Returns
        -------
            pd.Series
                timestamps referring to the start of the predictions.
            np.array
                array containing the forecasts
        """
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1

        # set start_date for first naive prediction
        start_date_index = self.min_observations
        dates = df["ds"].iloc[start_date_index : -self.n_forecasts + 1].reset_index(drop=True)
        # assemble last values based on K
        last_k_vals_arrays = [
            df["y"].iloc[i : i + self.K].values for i in range(0, df["y"].shape[0] - self.K - self.n_forecasts - 1)
        ]
        last_k_vals = np.stack(last_k_vals_arrays, axis=0)
        # Compute the predictions
        predicted = np.array([last_k_vals[:, i % self.K] for i in range(self.n_forecasts)]).T

        # No un-scaling and un-normalization needed. Components not applicable for naive model
        return dates, predicted

    def _reshape_raw_predictions_to_forecst_df(self, df_i, predicted):  # Todo outsource to df_utils?
        """Turns forecast-origin-wise predictions into forecast-target-wise predictions.
        Parameters
        ----------
            df : pd.DataFrame
                input dataframe
            predicted : np.array
                Array containing the forecasts
        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``,``ID`` and [``yhat<i>``],
                Note
                ----
                where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
                e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
        """
        cols = ["ds", "y", "ID"]  # cols to keep from df
        fcst_df = pd.concat((df_i[cols],), axis=1)
        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
        for forecast_lag in range(1, self.n_forecasts + 1):
            forecast = predicted[:, forecast_lag - 1]
            pad_before = self.min_observations + forecast_lag - 1
            pad_after = self.n_forecasts - forecast_lag
            yhat = np.concatenate(
                ([np.NaN] * pad_before, forecast, [np.NaN] * pad_after)
            )  # add pad based on n_forecasts and current forecast_lag
            name = f"yhat{forecast_lag}"
            fcst_df[name] = yhat

        return fcst_df
