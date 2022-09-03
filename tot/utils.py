import logging
import math

import numpy as np
import pandas as pd

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
