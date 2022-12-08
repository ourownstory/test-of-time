import logging
import math

import numpy as np
import pandas as pd

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
