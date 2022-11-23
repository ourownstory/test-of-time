import logging
import math

import numpy as np
import pandas as pd
from collections import OrderedDict
from neuralprophet import df_utils

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


def helper_tabularize_and_normalize(
    df,
    n_lags=0,
    n_forecasts=1,
):

    # normalize dataframe
    df, _ = df_utils.check_dataframe(df)
    df, _, _, _, _ = df_utils.prep_or_copy_df(df)
    _, global_data_params = df_utils.init_data_params(df=df, normalize="minmax")
    df = df_utils.normalize(df.copy(deep=True).drop("ID", axis=1), global_data_params)

    n_samples = len(df) - n_lags + 1 - n_forecasts
    # data is stored in OrderedDict
    inputs = OrderedDict({})

    def _stride_time_features_for_forecasts(x):
        # only for case where n_lags > 0
        return np.array([x[n_lags + i : n_lags + i + n_forecasts] for i in range(n_samples)], dtype=np.float64)

    # time is the time at each forecast step
    t = df.loc[:, "t"].values
    if n_lags == 0:
        assert n_forecasts == 1
        time = np.expand_dims(t, 1)
    else:
        time = _stride_time_features_for_forecasts(t)
    inputs["time"] = time

    def _stride_lagged_features(df_col_name, feature_dims):
        # only for case where n_lags > 0
        series = df.loc[:, df_col_name].values
        ## Added dtype=np.float64 to solve the problem with np.isnan for ubuntu test
        return np.array([series[i + n_lags - feature_dims : i + n_lags] for i in range(n_samples)], dtype=np.float64)

    if n_lags > 0 and "y" in df.columns:
        inputs["lags"] = _stride_lagged_features(df_col_name="y_scaled", feature_dims=n_lags)
        if np.isnan(inputs["lags"]).any():
            raise ValueError("Input lags contain NaN values in y.")

    targets = _stride_time_features_for_forecasts(df["y_scaled"].values)

    return inputs, targets, global_data_params


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
