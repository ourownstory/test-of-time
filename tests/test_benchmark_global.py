# !/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

from tot.benchmark import ManualBenchmark, SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.experiment import SimpleExperiment
from tot.models.models_neuralprophet import NeuralProphetModel

log = logging.getLogger("tot.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "datasets")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
ERCOT_FILE = os.path.join(DATA_DIR, "ercot_load_reduced.csv")
SAVE_DIR = os.path.join(DIR, "tests", "test-logs")
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

try:
    from prophet import Prophet

    _prophet_installed = True
except ImportError:
    Prophet = None
    _prophet_installed = False

NROWS = 200
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0
ERCOT_REGIONS = ["NORTH", "EAST", "FAR_WEST"]

PLOT = False


def test_benchmark_simple_global_modeling():
    # test NeuralProphetModel on all global model configurations
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
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 8,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "trend_global_local": "local",
                "season_global_local": "local",
                "global_normalization": False,
                "global_time_normalization": False,
            },
        ),
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 8,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": False,
                "global_time_normalization": True,
            },
        ),
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 8,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": True,
                "global_time_normalization": True,
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=["MAE", "MSE", "MASE", "RMSE"],
        test_percentage=0.25,
    )
    results_train, results_test = benchmark.run()

    log.debug("{}".format(results_test))
    print(results_test)


def test_benchmark_manual_global_modeling():
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
    peyton_manning_df_aux = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    peyton_manning_df = pd.DataFrame()
    slice_idx = 0
    for df_name in ["df1", "df2"]:
        df_aux = peyton_manning_df_aux.iloc[slice_idx : slice_idx + 100]
        df_aux = df_aux.assign(ID=df_name)
        peyton_manning_df = pd.concat((peyton_manning_df, df_aux), ignore_index=True)
        slice_idx = slice_idx + 100
    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={
                "n_lags": 5,
                "n_forecasts": 3,
                "epochs": EPOCHS,
                "learning_rate": 0.1,
            },
            data=Dataset(df=ercot_df, name="ercot_load", freq="H"),
            metrics=metrics,
            test_percentage=0.25,
        ),
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={
                "seasonality_mode": "multiplicative",
                "learning_rate": 0.1,
                "epochs": EPOCHS,
            },
            data=Dataset(df=peyton_manning_df, name="peyton_manning_many_ts", freq="D"),
            metrics=metrics,
            test_percentage=0.25,
        ),
    ]
    benchmark = ManualBenchmark(
        experiments=experiments,
        metrics=metrics,
        save_dir=SAVE_DIR,
    )
    results_train, results_test = benchmark.run()
    log.debug("{}".format(results_test))


def test_benchmark_dict_global_modeling():
    ercot_df = pd.read_csv(ERCOT_FILE)

    dataset_list = [
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 8,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": False,
                "global_time_normalization": True,
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE", "MASE", "RMSE"],
        test_percentage=0.25,
    )
    results_train, results_test = benchmark.run()

    log.debug("{}".format(results_test))
