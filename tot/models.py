import logging
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, df_utils

from tot.utils import _get_seasons, convert_to_datetime

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
        if historic data is used to fit the model (e.g. n_lags>0 or K>0): adds number of historic observations
        (e.g. n_lags or K) values to start of df_test.
        else (time-features only): returns unchanged df_test
        """
        return df_test.reset_index(drop=True)

    def maybe_drop_first_forecasts(self, predicted, df):
        """
        if historic data is used to fit the model (e.g. n_lags>0 or K>0): removes first n_lags values from
        predicted and df_test
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
        self.K = None

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
        self.K = None

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
    A `SeasonalNaiveModel` is a naive model that forecasts future values of a target series based on past observations
    of the target series of the specified period, i.e season.

    The naive models act on time series data having ``fit()`` and ``predict()`` methods. The season can be specified
    by K steps. Therefore, the model predicts the value of `K` time steps ago. When `K>1`, it repeats the last `K`
    values.

    Parameters
    ----------
        K : int
            the number of last time steps to repeat
        n_forecasts : int
            Number of steps ahead of prediction time step to forecast.
    Note
    ----
        ``Supported capabilities``
        * univariate time series
        * single target
        * manually select season steps K
        * n_forecats>1

        ``Not supported capabilities``
        * autoregression

        ``Planned capabilities``
        * auto-detect season
        * multi-variate time series only with multi-target
        * frequency check and optional frequency conversion
    """

    model_name: str = "SeasonalNaive"
    model_class: Type = None  # make sense?

    def __post_init__(self):
        # no installation checks required

        # expect no _data_params
        data_params = self.params["_data_params"]
        assert len(data_params) == 0
        custom_seasonalities = None  # TODO: auto-detect season from metadata

        # Verify expected model_prams: K > 1, n_forecasts >1
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        assert len(model_params) == 2, "Model parameters K and n_forecasts must be provided"
        self.n_forecasts = model_params["n_forecasts"]
        assert self.n_forecasts > 1, "Model parameter n_forecasts must be >1. "
        self.K = model_params["K"]  # for seasonal naive K is input parameter
        assert self.K > 1, "Model parameter K must be >1 for seasonal naive. For K=1 select NaiveModel instead. "
        self.n_lags = None  # TODO: should not be set to None. Find different solution.

    def fit(self, df: pd.DataFrame, freq: str):
        """Fits the naive model.
        Naive models do not need to be explicitly fitted. However, we store fitting-related information.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with all data
            freq : str
                frequency of the input data
        """
        df, received_ID_col, received_single_time_series, received_dict, _ = df_utils.prep_or_copy_df(df)
        # Receives df with single ID column. Only single time series accepted.
        assert len(df["ID"].unique()) == 1  # TODO: add multi-ID, multi-target

        self.freq = freq

    def predict(self, df: pd.DataFrame):
        """Runs the model to make predictions.
        Expects all data to be present in dataframe.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
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
        df, received_ID_col, received_single_time_series, received_dict, _ = df_utils.prep_or_copy_df(df)
        # Receives df with single ID column. Only single time series accepted.
        assert len(df["ID"].unique()) == 1  # TODO: add multi-ID, multi-target

        forecast = pd.DataFrame
        # check also no id column
        for df_name, df_i in df.groupby("ID"):
            dates, predicted = self._predict_raw(df_i)
            forecast = self._reshape_raw_predictions_to_forecst_df(df_i, predicted)
        fcst_df = df_utils.return_df_in_original_format(
            forecast, received_ID_col, received_single_time_series, received_dict
        )
        return fcst_df

    def maybe_add_first_inputs_to_df(self, df_train, df_test):
        """Adds last K values from df_train to start of df_test.

        Parameters
        ----------
            df_train: pd.DataFrame
                dataframe containing train data
            df_test: pd.DataFrame
                dataframe containing test data of previous split

        Returns
        -------
            pd.DataFrame
                dataframe containing test data enlarged with K values.
        """
        if self.K > 0:
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
                df_test_i = pd.concat([df_train_i.tail(self.K), df_test_i], ignore_index=True)
                df_test_new = pd.concat((df_test_new, df_test_i), ignore_index=True)
            df_test = df_utils.return_df_in_original_format(
                df_test_new, received_ID_col_test, received_single_time_series_test, received_dict_test
            )
        return df_test

    def maybe_drop_first_forecasts(self, predicted, df):
        """
        Removes first K values from predicted and df that have been previously added.

        Parameters
        ----------
            predicted: pd.DataFrame
                dataframe containing predicted data
            df: pd.DataFrame
                dataframe containing initial data

        Returns
        -------
            pd.DataFrame
                dataframe containing predicted data reduced by the first K values.
            pd.DataFrame
                dataframe containing initial data reduced by the first K values.
        """
        if self.K > 0:
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
                predicted_i = predicted_i[self.K :]
                df_i = df_i[self.K :]
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
        """if Model imputed any dates: removes any dates in predicted which are not in df_test.

        Parameters
        ----------
            predicted: pd.DataFrame
                dataframe containing predicted data
            df: pd.DataFrame
                dataframe containing initial data

        Returns
        -------
            pd.DataFrame
                dataframe containing predicted data of dates that are present in df.
            pd.DataFrame
                dataframe containing initial data.
        """
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
            predicted_i["ds"] = convert_to_datetime(predicted_i["ds"])
            df_i["ds"] = convert_to_datetime(df_i["ds"])
            df_i.set_index("ds", inplace=True)
            predicted_i.set_index("ds", inplace=True)
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

    def _predict_raw(self, df):
        """Computes forecast-origin-wise seasonal naive predictions.

        Predictions are returned in vector format. Predictions are given on a forecast origin basis,
        not on a target basis.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

        Returns
        -------
            pd.Series
                timestamps referring to the start of the predictions.
            np.array
                array containing the predictions
        """
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1

        dates = df["ds"].iloc[self.K : -self.n_forecasts + 1].reset_index(drop=True)
        # assemble last values based on K
        last_k_vals_arrays = [df["y"].iloc[i : i + self.K].values for i in range(0, dates.shape[0])]
        last_k_vals = np.stack(last_k_vals_arrays, axis=0)
        # Compute the predictions
        predicted = np.array([last_k_vals[:, i % self.K] for i in range(self.n_forecasts)]).T

        # No un-scaling and un-normalization needed. Operations not applicable for naive model
        return dates, predicted

    def _reshape_raw_predictions_to_forecst_df(self, df_i, predicted):  # Todo outsource to df_utils?
        """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

        Parameters
        ----------
            df : pd.DataFrame
                input dataframe
            predicted : np.array
                Array containing the predictions

        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``, optionally ``ID`` and [``yhat<i>``],

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
            pad_before = self.K + forecast_lag - 1
            pad_after = self.n_forecasts - forecast_lag
            yhat = np.concatenate(
                ([np.NaN] * pad_before, forecast, [np.NaN] * pad_after)
            )  # add pad based on n_forecasts and current forecast_lag
            name = f"yhat{forecast_lag}"
            fcst_df[name] = yhat

        return fcst_df
