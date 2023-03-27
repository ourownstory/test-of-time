#!/usr/bin/env python3
import logging
import os
import pathlib

import pandas as pd
import pytest

from tot.benchmark import SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.evaluation.metrics import ERROR_FUNCTIONS
from tot.models.models_neuralprophet import NeuralProphetModel, TorchProphetModel

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


NROWS = 128
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0
ERCOT_REGIONS = ["NORTH", "EAST", "FAR_WEST"]

PLOT = False


def test_lag_reg():
    log.info(f"testing: Add lagged regressors to models")
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    peyton_manning_df["A"] = peyton_manning_df["y"].rolling(7, min_periods=1).mean()
    peyton_manning_df["B"] = peyton_manning_df["y"].rolling(30, min_periods=1).mean()
    dataset_list = [
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {"n_lags": 3, "n_forecasts": 2, "epochs": 3, "lagged_regressors": {"A", "B"}},
        ),
        (
            NeuralProphetModel,
            {
                "n_lags": 3,
                "n_forecasts": 2,
                "epochs": 3,
                "lagged_regressors": {
                    "A": {"n_lags": 5, "regularization": 0.9, "normalize": False},
                    "B": {"n_lags": 5},
                },
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=0.25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_train, results_test = benchmark.run()

    print(results_test)

    # test for simple panel dataset
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
    ercot_df["A"] = ercot_df["y"].rolling(7, min_periods=1).mean()
    ercot_df["B"] = ercot_df["y"].rolling(30, min_periods=1).mean()
    dataset_list = [
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=0.25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_train, results_test = benchmark.run()
    log.info("#### done with test_lag_reg")
    print(results_test)


def test_future_reg():
    log.info(f"testing: Add future regressors to models")
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    peyton_manning_df["A"] = peyton_manning_df["y"].rolling(7, min_periods=1).mean()
    peyton_manning_df["B"] = peyton_manning_df["y"].rolling(30, min_periods=1).mean()
    dataset_list = [
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "n_lags": 3,
                "n_forecasts": 2,
                "epochs": 3,
                "future_regressors": {"A", "B"},
            },
        ),
        (
            NeuralProphetModel,
            {
                "n_lags": 3,
                "n_forecasts": 2,
                "epochs": 3,
                "future_regressors": {
                    "A": {"mode": "multiplicative", "regularization": 0.9, "normalize": "auto"},
                    "B": {"mode": "multiplicative"},
                },
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=0.25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_train, results_test = benchmark.run()
    print(results_test)

    # test for simple panel dataset
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
    ercot_df["A"] = ercot_df["y"].rolling(7, min_periods=1).mean()
    ercot_df["B"] = ercot_df["y"].rolling(30, min_periods=1).mean()
    dataset_list = [
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=0.25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_train, results_test = benchmark.run()
    log.info("#### done with test_future_reg")
    print(results_test)
