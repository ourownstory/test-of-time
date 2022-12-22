import logging

import numpy as np
import pandas as pd

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
