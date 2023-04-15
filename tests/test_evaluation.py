import logging
import os
import pathlib

import pandas as pd
import pytest

from tot.benchmark import SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.df_utils import prep_or_copy_df
from tot.evaluation.metric_utils import calculate_metrics_by_ID_for_forecast_step
from tot.evaluation.metrics import ERROR_FUNCTIONS
from tot.models.models_neuralprophet import NeuralProphetModel

log = logging.getLogger("tot.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "datasets")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
ERCOT_FILE = os.path.join(DATA_DIR, "ercot_load_reduced.csv")
EPOCHS = 1
BATCH_SIZE = 64
LR = 1.0
ERCOT_REGIONS = ["NORTH", "EAST", "FAR_WEST"]
NROWS = 128


def test_evaluation_by_ID_for_forecast_step():
    # select evaluation metrics
    metrics = ERROR_FUNCTIONS
    # select datasets
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    ercot_df_aux = pd.read_csv(ERCOT_FILE)
    ercot_df = pd.DataFrame()
    for region in ERCOT_REGIONS:
        ercot_df = pd.concat(
            (
                ercot_df,
                ercot_df_aux[ercot_df_aux["ID"] == region].iloc[:NROWS].copy(deep=True),
            ),
            ignore_index=True,
        )
    dataset_list = [
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 7,
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
        ),
    ]
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(metrics.keys()),
        test_percentage=0.25,
    )
    results_train, results_test = benchmark.run()
    # extract forecast dataframes
    fcst_test_peyton, _, _, _ = prep_or_copy_df(
        benchmark.fcst_test[0]
    )  # ensure ID column in dataframe with single time series
    fcst_train_peyton, _, _, _ = prep_or_copy_df(
        benchmark.fcst_train[0]
    )  # ensure ID column in dataframe with single time series
    fcst_test_ercot = benchmark.fcst_test[1]
    fcst_train_ercot = benchmark.fcst_train[1]
    # calculate metrics by ID for selected forecast step
    metrics_by_ID_yhat1_peyton = calculate_metrics_by_ID_for_forecast_step(
        fcst_df=fcst_test_peyton, df_historic=fcst_train_peyton, metrics=metrics, forecast_step_in_focus=1, freq="D"
    )
    metrics_by_ID_avg_yhat_peyton = calculate_metrics_by_ID_for_forecast_step(
        fcst_df=fcst_test_peyton, df_historic=fcst_train_peyton, metrics=metrics, forecast_step_in_focus=None, freq="D"
    )
    metrics_by_ID_yhat7_ercot = calculate_metrics_by_ID_for_forecast_step(
        fcst_df=fcst_test_ercot, df_historic=fcst_train_ercot, metrics=metrics, forecast_step_in_focus=1, freq="H"
    )

    log.debug("peyton_manning_df - metrics for yhat1: ", metrics_by_ID_yhat1_peyton)
    log.debug("peyton_manning_df - metrics average over all forecast steps: ", metrics_by_ID_avg_yhat_peyton)
    log.debug("ercot_df - metrics for yhat1: ", metrics_by_ID_yhat7_ercot)


def test_evaluation_by_ID_for_forecast_step_invalid_input():
    # select evaluation metrics
    metrics = ERROR_FUNCTIONS
    # select datasets
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 7,
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
        ),
    ]
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(metrics.keys()),
        test_percentage=0.25,
    )
    results_train, results_test = benchmark.run()
    # extract forecast dataframes
    fcst_test_peyton, _, _, _ = prep_or_copy_df(
        benchmark.fcst_test[0]
    )  # ensure ID column in dataframe with single time series
    fcst_train_peyton, _, _, _ = prep_or_copy_df(
        benchmark.fcst_train[0]
    )  # ensure ID column in dataframe with single time series
    # calculate metrics by ID for selected forecast step
    with pytest.raises(AssertionError):
        calculate_metrics_by_ID_for_forecast_step(
            fcst_df=fcst_test_peyton, df_historic=fcst_train_peyton, forecast_step_in_focus=1, freq="D"
        )
    with pytest.raises(AssertionError):
        calculate_metrics_by_ID_for_forecast_step(
            fcst_df=fcst_test_peyton,
            df_historic=fcst_train_peyton,
            metrics=metrics,
            forecast_step_in_focus=1,
        )
