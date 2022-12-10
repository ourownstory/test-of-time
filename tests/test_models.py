#!/usr/bin/env python3
import logging
import os
import pathlib

import pandas as pd
import pytest

from tot.benchmark import CrossValidationBenchmark, ManualBenchmark, ManualCVBenchmark, SimpleBenchmark
from tot.dataset import Dataset
from tot.experiment import CrossValidationExperiment, SimpleExperiment
from tot.metrics import ERROR_FUNCTIONS
from tot.models import LinearRegressionModel, NeuralProphetModel, ProphetModel

log = logging.getLogger("tot.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
ERCOT_FILE = os.path.join(DATA_DIR, "ercot_load.csv")
SAVE_DIR = os.path.join(DIR, "tests", "test-logs")
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


NROWS = 128
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0
ERCOT_REGIONS = ["NORTH", "EAST", "FAR_WEST"]

PLOT = False


def test_simple_benchmark_prophet():
    log.info("test_simple_benchmark")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (ProphetModel, {}),
    ]

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    if _prophet_installed:
        results_train, results_test = benchmark.run()
        log.debug(results_test.to_string())
    else:
        with pytest.raises(RuntimeError):
            results_train, results_test = benchmark.run()
    log.info("#### Done with test_simple_benchmark_prophet")


def test_prophet_for_global_modeling():
    ercot_df_aux = pd.read_csv(ERCOT_FILE)
    ercot_df = pd.DataFrame()
    for region in ERCOT_REGIONS:
        ercot_df = pd.concat(
            (ercot_df, ercot_df_aux[ercot_df_aux["ID"] == region].iloc[:NROWS].copy(deep=True)), ignore_index=True
        )
    dataset_list = [
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
    ]
    model_classes_and_params = [
        (ProphetModel, {}),
    ]

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    if _prophet_installed:
        with pytest.raises(NotImplementedError):
            results_train, results_test = benchmark.run()
    else:
        with pytest.raises(RuntimeError):
            results_train, results_test = benchmark.run()


def test_regression_model_module():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (
            LinearRegressionModel,
            {"n_lags": 12, "output_chunk_length": 1, "n_forecasts": 4},
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_train, results_test = benchmark.run()
    log.info("#### Done with test_simple_benchmark_prophet")
    print(results_test)
