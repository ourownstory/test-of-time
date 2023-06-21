import json
import os
from abc import ABCMeta

import pandas as pd
from pandas import Index
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from experiments.pipeline.helpers.scalers import LogTransformer
from tot.evaluation.metric_utils import calculate_metrics_by_ID_for_forecast_step


def save_results(benchmark, metrics, freq, dir, save):
    if save:
        # result details
        for i, (fcst_test, fcst_train, exp) in enumerate(
            zip(benchmark.fcst_test, benchmark.fcst_train, benchmark.experiments)
        ):
            df_metric_perID = calculate_metrics_by_ID_for_forecast_step(
                fcst_df=fcst_test,
                df_historic=fcst_train,
                metrics=metrics,
                forecast_step_in_focus=None,
                freq=freq,
            )
            df_metric_perID.index = df_metric_perID.index.astype(str)
            df_metrics_sum = pd.DataFrame(benchmark.df_metrics_test.loc[[i], metrics])
            df_metrics_sum.index = Index(["ALL"], name="ID")
            df_metric_perID = pd.concat([df_metric_perID, df_metrics_sum], axis=0)

            file_name = os.path.join(dir, f"results_{exp.experiment_name}.csv")
            df_metric_perID.to_csv(file_name)

        # result summary
        metric_dfs = benchmark.df_metrics_test
        elapsed_times = benchmark.elapsed_times
        results_file_name = os.path.join(dir, "results.csv")
        metric_dfs.to_csv(results_file_name)

        df_elapsed_time = pd.DataFrame(elapsed_times, columns=["elapsed_times"])
        elapsed_time_file_name = os.path.join(dir, "elapsed_time.csv")
        df_elapsed_time.to_csv(elapsed_time_file_name)


def save_params(params, dir_name, save=True):
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, ABCMeta):
                return obj.__name__
            if isinstance(
                obj,
                (
                    StandardScaler,
                    MinMaxScaler,
                    RobustScaler,
                    LogTransformer,
                    PowerTransformer,
                    QuantileTransformer,
                    FunctionTransformer,
                ),
            ):
                return str(obj)
            return super().default(obj)

    if save:
        config_file_name = os.path.join(dir_name, "params.json")
        with open(config_file_name, "w") as file:
            json.dump(params, file, cls=CustomJSONEncoder)


def build_synth_data_name(
    data_func,
    params_name,
    n_ts_groups,
    amplitude_per_group,
    offset_per_group,
    data_trend_gradient_per_group,
    proportion_break,
):
    return "{}_{}_n_ts_{}_am_{}_of_{}_gr_{}_{}".format(
        data_func,
        params_name,
        n_ts_groups,
        amplitude_per_group,
        offset_per_group,
        data_trend_gradient_per_group,
        proportion_break,
    )


def build_real_data_name(
    dataset_name,
    gen_fun_name,
    params_name,
):
    return "{}_{}_{}".format(
        dataset_name,
        gen_fun_name,
        params_name,
    )
