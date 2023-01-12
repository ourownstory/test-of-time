import logging

import numpy as np
import pandas as pd
from neuralprophet.df_utils import prep_or_copy_df

log = logging.getLogger("tot.df_utils")


def reshape_raw_predictions_to_forecast_df(df, predicted, n_req_past_observations, n_req_future_observations):
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
    fcst_df = pd.concat((df[cols],), axis=1)
    # create a line for each forecast_lag
    # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
    for forecast_lag in range(1, n_req_future_observations + 1):
        forecast = predicted[:, forecast_lag - 1]
        pad_before = n_req_past_observations + forecast_lag - 1
        pad_after = n_req_future_observations - forecast_lag
        yhat = np.concatenate(
            ([np.NaN] * pad_before, forecast, [np.NaN] * pad_after)
        )  # add pad based on n_forecasts and current forecast_lag
        name = f"yhat{forecast_lag}"
        fcst_df[name] = yhat

    return fcst_df


def _split_df(df, n_lags, test_percentage):
    """Splits timeseries df into train and validation sets.

    Parameters
    ----------
        df : pd.DataFrame
            data to be splitted
        n_lags : int
            lags, identical to NeuralProphet
        test_percentage : float, int

    Returns
    -------
        pd.DataFrame
            training data
        pd.DataFrame
            validation data
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    n_samples = len(df)
    if 0.0 < test_percentage < 1.0:
        n_valid = max(1, int(n_samples * test_percentage))
    else:
        assert test_percentage >= 1
        assert type(test_percentage) == int
        n_valid = test_percentage
    n_train = n_samples - n_valid
    assert n_train >= 1

    split_idx_train = n_train
    split_idx_val = split_idx_train + 1 if n_lags == 0 else split_idx_train - (n_lags + 1) + 1
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    log.debug(f"{n_train} n_train, {n_samples - n_train} n_eval")
    return df_train, df_val


def split_df(df, n_lags, test_percentage=0.25, local_split=False):
    """Splits timeseries df into train and validation sets.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        n_lags : int
            lags, identical to NeuralProphet
        test_percentage : float, int
            fraction (0,1) of data to use for holdout validation set, or number of validation samples >1

    Returns
    -------
        pd.DataFrame
            training data
        pd.DataFrame
            validation data
    """
    df, _, _, _ = prep_or_copy_df(df)

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    if local_split:
        for df_name, df_i in df.groupby("ID"):
            df_t, df_v = _split_df(df_i, n_lags, test_percentage)
            df_train = pd.concat((df_train, df_t.copy(deep=True)), ignore_index=True)
            df_val = pd.concat((df_val, df_v.copy(deep=True)), ignore_index=True)
    else:
        if len(df["ID"].unique()) == 1:
            for df_name, df_i in df.groupby("ID"):
                df_train, df_val = _split_df(df_i, n_lags, test_percentage)

    # df_train and df_val are returned as pd.DataFrames
    return df_train, df_val
