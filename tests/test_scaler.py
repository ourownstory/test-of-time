#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
from sklearn.preprocessing import StandardScaler

from tot.benchmark import SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.models.models_neuralprophet import NeuralProphetModel

log = logging.getLogger("tot.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "datasets")
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

NROWS = 128
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0
ERCOT_REGIONS = ["NORTH", "EAST", "FAR_WEST"]

PLOT = False


def test_scaling_per_dataset():
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
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)

    dataset_list = [
        Dataset(
            df=air_passengers_df,
            name="air_passengers",
            freq="MS",
            seasonality_mode="multiplicative",
        ),
        Dataset(
            df=ercot_df,
            name="ercot",
            freq="H",
        ),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "scaler": StandardScaler(),
                "scaling_level": "per_dataset",
                "n_lags": 5,
                "n_forecasts": 3,
                "learning_rate": 0.1,
                "normalize": "off",
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=["MAE"],
        test_percentage=0.25,
    )
    results_train, results_test = benchmark.run()

    log.debug("{}".format(results_test))


def test_scaling_per_time_series():
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
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)

    dataset_list = [
        Dataset(
            df=air_passengers_df,
            name="air_passengers",
            freq="MS",
            seasonality_mode="multiplicative",
        ),
        Dataset(
            df=ercot_df,
            name="ercot",
            freq="H",
        ),
    ]
    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "scaler": StandardScaler(),
                "scaling_level": "per_time_series",
                "n_lags": 5,
                "n_forecasts": 3,
                "learning_rate": 0.1,
                "normalize": "off",
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=["MAE"],
        test_percentage=0.25,
    )
    results_train, results_test = benchmark.run()

    log.debug("{}".format(results_test))
