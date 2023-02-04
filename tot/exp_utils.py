import logging

import numpy as np

from tot.metrics import ERROR_FUNCTIONS

log = logging.getLogger("tot.exp_utils")


def evaluate_forecast(fcst_train, fcst_test, metrics, metadata=None):
    """
    Evaluate forecast performance on training and test data using specified metrics.

    Parameters
    ----------
    fcst_train : pandas.DataFrame
        Forecast data on the training set.
    fcst_test : pandas.DataFrame
        Forecast data on the test set.
    metrics : list
        List of error metrics to evaluate the forecast.
    metadata : dict
        Metadata to be stored in the results.

    Returns
    -------
    result_train : pandas.DataFrame
        Result of evaluation on training set.
    result_test : pandas.DataFrame
        Result of evaluation on test set.
    """

    if metadata is not None:
        result_train = metadata.copy()
        result_test = metadata.copy()
    else:
        result_train = {}
        result_test = {}

    for metric in metrics:
        # todo: parallelize
        yhat_train = [col for col in fcst_train.columns if "yhat" in col]
        yhat_test = [col for col in fcst_test.columns if "yhat" in col]
        n_yhats_train = len(yhat_train)
        n_yhats_test = len(yhat_test)

        assert n_yhats_train == n_yhats_test, "Dimensions of fcst dataframe faulty."

        metric_train_list = []
        metric_test_list = []

        fcst_train = fcst_train.fillna(value=np.nan)
        fcst_test = fcst_test.fillna(value=np.nan)

        for x in range(1, n_yhats_train + 1):
            metric_train_list.append(
                ERROR_FUNCTIONS[metric](
                    predictions=fcst_train["yhat{}".format(x)].values,
                    truth=fcst_train["y"].values,
                    truth_train=fcst_train["y"].values,
                )
            )
            metric_test_list.append(
                ERROR_FUNCTIONS[metric](
                    predictions=fcst_test["yhat{}".format(x)].values,
                    truth=fcst_test["y"].values,
                    truth_train=fcst_train["y"].values,
                )
            )
        result_train[metric] = np.nanmean(metric_train_list, dtype="float32")
        result_test[metric] = np.nanmean(metric_test_list, dtype="float32")

    return result_train, result_test
