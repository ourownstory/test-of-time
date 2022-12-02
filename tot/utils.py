import logging
import math

import numpy as np
import pandas as pd
from darts import TimeSeries

log = logging.getLogger("tot.util")


def convert_to_datetime(series):
    if series.isnull().any():
        raise ValueError("Found NaN in column ds.")
    if series.dtype == np.int64:
        series = series.astype(str)
    if not np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series)
    if series.dt.tz is not None:
        raise ValueError("Column ds has timezone specified, which is not supported. Remove timezone.")
    return series


def _get_seasons(seasonalities):
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
