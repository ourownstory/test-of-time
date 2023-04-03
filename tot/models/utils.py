import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries

from tot.df_utils import _validate_single_ID_df

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


def reshape_raw_predictions_to_forecast_df(
    df, predicted, past_observations_per_prediction, future_observations_per_prediction
):
    """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

    Parameters
    ----------
        df : pd.DataFrame
            input dataframe
        predicted : np.array
            Array containing the predictions
        past_observations_per_prediction: int
            past observation samples required for one prediction step
        future_observations_per_prediction: int
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

    _validate_single_ID_df(df)

    cols = ["ds", "y", "ID"]  # cols to keep from df
    fcst_df = pd.concat((df[cols],), axis=1)
    # create a line for each forecast_lag
    # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
    for fcst_sample in range(1, future_observations_per_prediction + 1):
        forecast = predicted[:, fcst_sample - 1]
        pad_before = past_observations_per_prediction + fcst_sample - 1
        pad_after = future_observations_per_prediction - fcst_sample
        yhat = np.concatenate(
            ([np.NaN] * pad_before, forecast, [np.NaN] * pad_after)
        )  # add pad based on n_forecasts and current forecast_lag
        name = f"yhat{fcst_sample}"
        fcst_df[name] = yhat

    return fcst_df


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


def convert_df_to_TimeSeries(df, freq) -> TimeSeries:
    """
    Converts pd.Dataframe to TimeSeries (e.g. output of darts).

    Parameters
    ----------
        df : pd.Dataframe
            time series to be fitted or predicted
        freq : str (must ahere to darts format)
            Optionally, a string representing the frequency of the Pandas DateTimeIndex.

    Returns
    ----------
        series : TimeSeries
            time series to be fitted or predicted

    """
    received_single_ts = len(df["ID"].unique()) == 1

    if not received_single_ts:
        # pivot the resulting DataFrame to convert multiple time series
        df = df.pivot(index="ds", columns="ID", values="y").rename_axis(columns=None).reset_index()
        value_cols = df.columns[1:].tolist()
    else:
        value_cols = "y"
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
            past_observations_per_prediction=season_length,
            future_observations_per_prediction=n_forecasts,
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
    _validate_single_ID_df(df)

    dates = df["ds"].iloc[season_length : -n_forecasts + 1].reset_index(drop=True)
    # assemble last values based on season_length
    last_k_vals_arrays = [df["y"].iloc[i : i + season_length].values for i in range(0, dates.shape[0])]
    last_k_vals = np.stack(last_k_vals_arrays, axis=0)
    # Compute the predictions
    predicted = np.array([last_k_vals[:, i % season_length] for i in range(n_forecasts)]).T
    if predicted.shape[1] is None:
        predicted = predicted.reshape(predicted.shape[0], 1)

    # No un-scaling and un-normalization needed. Operations not applicable for naive model
    return predicted


def _predict_darts_model(
    df,
    model,
    past_observations_per_prediction,
    future_observations_per_prediction,
    retrain,
    received_single_time_series,
):
    """Computes forecast-target-wise predictions for the passed darts model.

    Parameters
    ----------
        model : LinearRegressionModel
            model to be predicted
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and ``ID`` with all data
        past_observations_per_prediction : int
            number of past observations needed for prediction
        future_observations_per_prediction : int
            number of future samples to be predicted in one step
        retrain : bool
            flag specific to darts models that indicates whether the retrain mode is activated
        received_single_time_series : bool
            whether it is a single time series

    Returns
    -------
        pd.DataFrame
            columns ``ds``, ``y``, optionally ``ID`` and [``yhat<i>``],

            Note
            ----
            where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
            e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
    """
    predicted = _predict_raw_darts_model(
        df=df,
        model=model,
        past_observations_per_prediction=past_observations_per_prediction,
        future_observations_per_prediction=future_observations_per_prediction,
        retrain=retrain,
        received_single_time_series=received_single_time_series,
    )
    fcst_df = (
        df.groupby("ID")
        .apply(
            lambda x: reshape_raw_predictions_to_forecast_df(
                x,
                predicted[x.name],
                past_observations_per_prediction=past_observations_per_prediction,
                future_observations_per_prediction=future_observations_per_prediction,
            )
        )
        .reset_index(drop=True)
    )

    return fcst_df


def _predict_raw_darts_model(
    df,
    model,
    past_observations_per_prediction,
    future_observations_per_prediction,
    retrain,
    received_single_time_series,
):
    """Computes forecast-origin-wise predictions for the passed darts model for single time series.
    Predictions are returned in vector format. Predictions are given on a forecast origin basis,
    not on a target basis.
    Parameters
    ----------
        model : LinearRegressionModel
            model to be predicted
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        past_observations_per_prediction : int
            number of past observations needed for prediction
        future_observations_per_prediction : int
            number of future samples to be predicted in one step
        retrain : bool
            flag specific to darts models that indicates whether the retrain mode is activated
        received_single_time_series : bool
            flag that specific if the dataframe has a single or multiple time series

    Returns
    -------
        np.array
            array containing the predictions
    """
    series = convert_df_to_TimeSeries(df, freq=model.freq)
    predicted_list = model.model.historical_forecasts(
        series,
        start=past_observations_per_prediction,
        forecast_horizon=future_observations_per_prediction,
        retrain=retrain,
        last_points_only=False,
        verbose=True,
    )
    # Convert (list of) TimeSeries to np.array
    predicted = np.array([prediction_series.values() for prediction_series in predicted_list])
    prediction_ids = ["__df__"] if received_single_time_series else predicted_list[0].components
    predicted = np.transpose(predicted, (2, 0, 1))
    predicted_dict = {prediction_ids[id]: predicted[id, :, :] for id in range(predicted.shape[0])}
    # Return dict with array per ID
    return predicted_dict
