import logging
import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from darts import TimeSeries

log = logging.getLogger("tot.utils")

FREQ_TO_SEASON_STEP_MAPPING = {
    "MS": {"yearly": 12},
    "M": {"yearly": 12},
    "YS": {"custom": 1},
    "DS": {"yearly": 356, "weekly": 7, "daily": 1},
    "D": {"yearly": 356, "weekly": 7, "daily": 1},
    "HS": {"yearly": 24 * 356, "weekly": 24 * 7, "daily": 24},
    "H": {"yearly": 24 * 356, "weekly": 24 * 7, "daily": 24},
    "5min": {"yearly": 12 * 24 * 356, "weekly": 12 * 24 * 7, "daily": 12 * 24},
}


def reshape_raw_predictions_to_forecast_df(df, predicted, n_req_past_obs, n_req_future_obs):
    """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

    Parameters
    ----------
        df : pd.DataFrame
            input dataframe
        predicted : np.array
            Array containing the predictions
        n_req_past_obs: int
            past observation samples required for one prediction step
        n_req_future_obs: int
            future observation samples required for one prediction step

    Returns
    -------
        pd.DataFrame
            columns ``ds``, ``y``, optionally ``ID`` and [``yhat<i>``],

            Note
            ----
            where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
            e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
    """
    assert len(df["ID"].unique()) == 1
    cols = ["ds", "y", "ID"]  # cols to keep from df
    fcst_df = pd.concat((df[cols],), axis=1)
    # create a line for each forecast_lag
    # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
    for fcst_sample in range(1, n_req_future_obs + 1):
        forecast = predicted[:, fcst_sample - 1]
        pad_before = n_req_past_obs + fcst_sample - 1
        pad_after = n_req_future_obs - fcst_sample
        yhat = np.concatenate(
            ([np.NaN] * pad_before, forecast, [np.NaN] * pad_after)
        )  # add pad based on n_forecasts and current forecast_lag
        name = f"yhat{fcst_sample}"
        fcst_df[name] = yhat

    return fcst_df


def convert_to_datetime(series: pd.Series) -> pd.Series:
    """Convert input series to datetime format

    Parameters
    ----------
        series : pd.Series
            input series that needs to be converted to datetime format

    Returns
    -------
        pd.Series
            series in datetime format

    Raises
    ------
        ValueError
            if input series contains NaN values or has timezone specified
    """
    if series.isnull().any():
        raise ValueError("Found NaN in column ds.")
    if series.dtype == np.int64:
        series = series.astype(str)
    if not np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series)
    if series.dt.tz is not None:
        raise ValueError("Column ds has timezone specified, which is not supported. Remove timezone.")
    return series


def _get_seasons(seasonalities: List[float]) -> Tuple[bool, bool, bool, List[float]]:
    """
    Given a list of seasons, returns a tuple of bools indicating if daily, weekly, and yearly seasons are present,
    and a list of custom seasons.
    Seasonality is considered daily if it is close to 1, weekly if it is close to 7, and yearly if it is close to 365.

    Parameters
    ----------
    seasonalities : List[float]
        List of seasonality in days

    Returns
    -------
    Tuple[bool, bool, bool, List[float]]
        Tuple of bools indicating if daily, weekly, and yearly seasons are present,
        and a list of custom seasons.
    """
    custom = []
    daily = False
    weekly = False
    yearly = False
    for season_days in seasonalities:
        if math.isclose(season_days, 1):
            daily = True
        elif math.isclose(season_days, 7):
            weekly = True
        elif math.isclose(season_days, 365) or math.isclose(season_days, 365.25):
            yearly = True
        else:
            custom.append(season_days)
    return daily, weekly, yearly, custom


def convert_df_to_TimeSeries(df, value_cols, freq) -> TimeSeries:
    """
    Converts pd.Dataframe to TimeSeries (e.g. output of darts).

    Parameters
    ----------
        df : pd.Dataframe
            time series to be fitted or predicted
        value_cols : List
            A string or list of strings representing the value column(s) to be extracted from the DataFrame.
        freq : str (must ahere to darts format)
            Optionally, a string representing the frequency of the Pandas DateTimeIndex.

    Returns
    ----------
        series : TimeSeries
            time series to be fitted or predicted

    """
    series = TimeSeries.from_dataframe(df=df, time_col="ds", value_cols=value_cols, freq=freq)
    return series


def _convert_seasonality_to_season_length(freq, daily=False, weekly=False, yearly=False, custom_seasonalities=None):
    """Convert seasonality to a number of time steps (season_length) for the given frequency.

    Parameters
    ----------
    freq: str
        The frequency to convert.
    daily: bool, optional
        Whether to use the daily season length. Defaults to False.
    weekly: bool, optional
        Whether to use the weekly season length. Defaults to False.
    yearly: bool, optional
        Whether to use the yearly season length. Defaults to False.
    custom_seasonalities: int or array-like, optional
        The custom season length to use. If provided, overrides the other season length arguments.

    Returns
    -------
    season_length: int
        The season length in number of time steps corresponding to the given frequency.
    """
    conversion_dict = FREQ_TO_SEASON_STEP_MAPPING[freq]
    season_length = None
    # assign smallest seasonality
    if custom_seasonalities:
        season_length = custom_seasonalities
    elif daily:
        season_length = conversion_dict["daily"]
    elif weekly:
        season_length = conversion_dict["weekly"]
    elif yearly:
        season_length = conversion_dict["yearly"]

    return season_length


def _predict_seasonal_naive(df, season_length, n_forecasts):
    """Computes forecast-target-wise seasonal naive predictions.

    Parameters
    ----------
        df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and ``ID`` with all data
        season_length : int
                seasonal period in number of time steps
        n_forecasts : int
                number of steps ahead of prediction time step to forecast

    Returns
    -------
        pd.DataFrame
            columns ``ds``, ``y``, optionally ``ID`` and [``yhat<i>``],

            Note
            ----
            where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
            e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
    """
    forecast_new = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        predicted_i = _predict_single_raw_seasonal_naive(df_i, season_length=season_length, n_forecasts=n_forecasts)
        forecast_i = reshape_raw_predictions_to_forecast_df(
            df_i,
            predicted_i,
            n_req_past_obs=season_length,
            n_req_future_obs=n_forecasts,
        )
        forecast_new = pd.concat((forecast_new, forecast_i), ignore_index=True)

    return forecast_new


def _predict_single_raw_seasonal_naive(df, season_length, n_forecasts):
    """Computes forecast-origin-wise seasonal naive predictions.
    Predictions are returned in vector format. Predictions are given on a forecast origin basis,
    not on a target basis.
    Parameters
    ----------
        df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and ``ID`` with all data
        season_length : int
                seasonal period in number of time steps
        n_forecasts : int
                number of steps ahead of prediction time step to forecast

    Returns
    -------
        pd.Series
            timestamps referring to the start of the predictions.
        np.array
            array containing the predictions
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1

    dates = df["ds"].iloc[season_length : -n_forecasts + 1].reset_index(drop=True)
    # assemble last values based on season_length
    last_k_vals_arrays = [df["y"].iloc[i : i + season_length].values for i in range(0, dates.shape[0])]
    last_k_vals = np.stack(last_k_vals_arrays, axis=0)
    # Compute the predictions
    predicted = np.array([last_k_vals[:, i % season_length] for i in range(n_forecasts)]).T

    # No un-scaling and un-normalization needed. Operations not applicable for naive model
    return predicted


def _predict_linear_regression(model, df):
    """Computes forecast-target-wise predictions for linear regression.

    Parameters
    ----------
        model : LinearRegressionModel
            model to be predicted
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and ``ID`` with all data
    Returns
    -------
        pd.DataFrame
            columns ``ds``, ``y``, optionally ``ID`` and [``yhat<i>``],

            Note
            ----
            where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
            e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
    """
    forecast_new = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        predicted_i = _predict_single_raw_linear_regression(model=model, df=df_i)
        forecast_i = reshape_raw_predictions_to_forecast_df(
            df_i,
            predicted_i,
            n_req_past_obs=model.n_lags,
            n_req_future_obs=model.n_forecasts,
        )
        forecast_new = pd.concat((forecast_new, forecast_i), ignore_index=True)

    return forecast_new


def _predict_single_raw_linear_regression(model, df):
    """Computes forecast-origin-wise predictions for linear regression for single time series.
    Predictions are returned in vector format. Predictions are given on a forecast origin basis,
    not on a target basis.
    Parameters
    ----------
        model : LinearRegressionModel
            model to be predicted
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
    Returns
    -------
        np.array
            array containing the predictions
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1

    value_cols = df.columns.values[1:-1].tolist()
    series = convert_df_to_TimeSeries(df, value_cols=value_cols, freq=model.freq)
    predicted_list = model.model.historical_forecasts(
        series,
        start=model.n_lags,
        forecast_horizon=model.n_forecasts,
        retrain=False,
        last_points_only=False,
        verbose=True,
    )
    # convert TimeSeries to np.array
    prediction_series = [prediction_series.values() for i, prediction_series in enumerate(predicted_list)]
    predicted = np.stack(prediction_series, axis=0).squeeze()

    # No un-scaling and un-normalization needed. Operations not applicable for naive model
    return predicted
