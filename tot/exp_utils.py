import logging
import math
from typing import List, Tuple

import numpy as np

from tot.metrics import ERROR_FUNCTIONS

log = logging.getLogger("tot.exp_utils")


def evaluate_forecast(fcst_train, fcst_test, metrics, metadata=None):
    result_train = metadata.copy()
    result_test = metadata.copy()

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
