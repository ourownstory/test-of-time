from typing import Optional

import numpy as np
import pandas as pd

from tot.evaluation.metrics import ERROR_FUNCTIONS


def calc_metrics_for_all_IDs_for_fcst_step(
    fcst_df: pd.DataFrame, metrics: list, forecast_step_in_focus: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculates the specified metrics for every ID and for a specific forecast step. If no forecast step is pspecified, it averages over all forecast steps.

    Parameters:
    -----------
    fcst_df : pd.DataFrame
        A DataFrame containing the forecast and an 'ID' column.
    metrics : list of str
        A list of metrics to calculate.
    forecast_step_in_focus : int or None (default None)
        The specific forecast step to calculate the metrics for. If None, the function
        averages the metrics over all forecast steps.

    Returns:
    --------
    metrics_df_per_forecast_step : pandas DataFrame
        A DataFrame containing the specified metrics for every ID and for a specific forecast step (if specified).

    Examples:
    ---------
    >>> fcst_df = pd.DataFrame({'ID': ["EAST", "EAST", "NORTH", "NORTH"],
    ...                         'y': [10, 20, 30, 40],
    ...                         'yhat1': [12, 22, 32, 42],
    ...                         'yhat2': [11, 21, 31, 41]})
    >>> metrics = ['MSE', 'MAE']
    >>> calc_metrics_for_all_IDs_for_fcst_step(fcst_df, metrics, forecast_step_in_focus=1)
       mse       mae
    ID
    1   4.0       2.0
    2   4.0       4.0
    >>> calc_metrics_for_all_IDs_for_fcst_step(fcst_df, metrics)
           mse       mae
    ID
    1  4.666667  2.666667
    2  4.666667  4.666667
    """
    # calculate the specified metrics for every ID and every forecast step
    metrics_df_all_IDs = fcst_df.groupby("ID").apply(
        lambda x: _calc_metrics_for_single_ID_and_every_fcst_step(x, metrics)
    )
    if forecast_step_in_focus is None:
        # select all metrics for all IDs and average over all forecast steps
        metrics_df_per_forecast_step = metrics_df_all_IDs.groupby("ID").apply(lambda x: x.mean(axis=0))
    else:
        # select all metrics for all IDs and the specified forecast step
        metrics_df_per_forecast_step = metrics_df_all_IDs[
            metrics_df_all_IDs.index.get_level_values(1).str.contains("yhat{}".format(str(forecast_step_in_focus)))
        ]
    return metrics_df_per_forecast_step


def _calc_metrics_for_single_ID_and_every_fcst_step(fcst_df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Calculates specified metrics for a single ID and every forecast step.

    Parameters:
    -----------
    fcst_df : pd.DataFrame
        The input dataframe containing y and yhat columns for a single ID.
    metrics : list of str
        The list of metric names to be calculated.

    Returns:
    --------
    metrics_df : pandas DataFrame
        The dataframe containing the calculated metrics for each forecast step.
    """
    # identify yhat columns
    yhat = [col for col in fcst_df.columns if "yhat" in col]
    fcst_df = fcst_df.fillna(value=np.nan)
    # calculate the specified metrics for every forecast step
    metrics_df = pd.concat(
        [
            fcst_df[yhat].apply(
                lambda x: ERROR_FUNCTIONS[metric](
                    predictions=x.values,
                    truth=fcst_df["y"].values,
                    truth_train=fcst_df["y"].values,
                )
            )
            for metric in metrics
        ],
        axis=1,
        keys=metrics,
    )

    return metrics_df


def calc_averaged_metrics_per_experiment(fcst_df: pd.DataFrame, metrics: list, metadata: Optional[dict] = None) -> dict:
    """
    Calculate the average of specified metrics over every ID and every forecast step.

    Parameters:
    -----------
    fcst_df : pandas.DataFrame
        A DataFrame containing the forecast data for each ID.
    metrics : list of str
        A list of metric names or a single metric name to calculate.
    metadata : dict or None, optional (default=None)
        Metadata to be included in the results.

    Returns:
    --------
    dict
        A dictionary containing the averaged metrics for all IDs and forecast steps.

    """
    if metadata is not None:
        metrics_results = metadata.copy()
    else:
        metrics_results = {}
    # calculate the specified metrics for every ID and every forecast step
    metrics_df_all_IDs = fcst_df.groupby("ID").apply(
        lambda x: _calc_metrics_for_single_ID_and_every_fcst_step(x, metrics)
    )

    # for each specified metric average here over all IDs and forecast steps
    yhat_average = metrics_df_all_IDs.mean(axis=0)
    # add to the metrics_results dict
    metrics_results.update(yhat_average.to_dict())

    return metrics_results
