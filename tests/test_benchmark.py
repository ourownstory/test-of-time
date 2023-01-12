#!/usr/bin/env python3

import logging
import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tot.benchmark import (
    CrossValidationBenchmark,
    ManualBenchmark,
    ManualCVBenchmark,
    SimpleBenchmark,
)
from tot.dataset import Dataset
from tot.experiment import CrossValidationExperiment, SimpleExperiment
from tot.metrics import ERROR_FUNCTIONS
from tot.models import NeuralProphetModel, ProphetModel

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


def test_2_benchmark_simple():
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)

    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS", seasonality_mode="multiplicative"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D", seasonalities=[7, 365.25]),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"n_lags": 5, "n_forecasts": 3, "learning_rate": 0.1, "epochs": EPOCHS}),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE", "MASE", "RMSE"],
        test_percentage=25,
    )
    results_train, results_test = benchmark.run()

    log.debug("{}".format(results_test))


def test_2_benchmark_CV():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"n_lags": 5, "n_forecasts": 3, "learning_rate": 0.1, "epochs": EPOCHS}),
        # (ProphetModel, {}), # needs to be installed
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark_cv = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MASE", "RMSE"],
        test_percentage=10,
        num_folds=3,
        fold_overlap_pct=0,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug("{}".format(results_summary))
    if PLOT:
        # model plot
        # air_passengers = results_summary[results_summary['data'] == 'air_passengers']
        # air_passengers = air_passengers[air_passengers['split'] == 'test']
        # plt_air = air_passengers.plot(x='model', y='RMSE', kind='barh')
        # data plot
        air_passengers = results_summary[results_summary["split"] == "test"]
        plt_air = air_passengers.plot(x="data", y="MASE", kind="barh")
        plt.show()


def test_2_benchmark_manual():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"n_lags": 5, "n_forecasts": 3, "epochs": EPOCHS, "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
        ),
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1, "epochs": EPOCHS},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
        ),
    ]
    if _prophet_installed:
        experiments.append(
            SimpleExperiment(
                model_class=ProphetModel,
                params={
                    "seasonality_mode": "multiplicative",
                },
                data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
                metrics=metrics,
                test_percentage=25,
            )
        )
    benchmark = ManualBenchmark(
        experiments=experiments,
        metrics=metrics,
        save_dir=SAVE_DIR,
    )
    results_train, results_test = benchmark.run()
    log.debug("{}".format(results_test))


def test_2_benchmark_manualCV():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"n_lags": 5, "n_forecasts": 3, "epochs": EPOCHS, "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS, "seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
        ),
        # needs to be installed
        # CrossValidationExperiment(
        #     model_class=ProphetModel,
        #     params={"seasonality_mode": "multiplicative", },
        #     data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        #     metrics=metrics,
        #     test_percentage=10,
        #     num_folds=3,
        #     fold_overlap_pct=0,
        # ),
    ]
    benchmark_cv = ManualCVBenchmark(
        experiments=experiments,
        metrics=metrics,
        save_dir=SAVE_DIR,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug("{}".format(results_summary))


def test_manual_benchmark():
    log.info("test_manual_benchmark")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    metrics = list(ERROR_FUNCTIONS.keys())
    experiments = [
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1, "epochs": EPOCHS},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
            save_dir=SAVE_DIR,
        ),
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"learning_rate": 0.1, "epochs": EPOCHS},
            data=Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
            metrics=metrics,
            test_percentage=15,
            save_dir=SAVE_DIR,
        ),
    ]
    prophet_exps = [
        SimpleExperiment(
            model_class=ProphetModel,
            params={"seasonality_mode": "multiplicative"},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
            save_dir=SAVE_DIR,
        ),
        SimpleExperiment(
            model_class=ProphetModel,
            params={},
            data=Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
            metrics=metrics,
            test_percentage=15,
            save_dir=SAVE_DIR,
        ),
    ]
    if _prophet_installed:
        experiments += prophet_exps
    benchmark = ManualBenchmark(experiments=experiments, metrics=metrics, num_processes=1)
    results_train, results_test = benchmark.run()
    log.debug(results_test.to_string())
    log.info("#### Done with test_manual_benchmark")


def test_manual_cv_benchmark():
    log.info("test_manual_cv_benchmark")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    metrics = list(ERROR_FUNCTIONS.keys())
    experiments = [
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS, "seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=2,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "epochs": EPOCHS, "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=1,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
        ),
    ]
    benchmark_cv = ManualCVBenchmark(experiments=experiments, metrics=metrics, num_processes=1)
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug(results_summary.to_string())
    log.debug(results_train.to_string())
    log.debug(results_test.to_string())
    log.info("#### Done with test_manual_cv_benchmark")


def test_simple_benchmark():
    log.info("test_simple_benchmark")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
        # Dataset(df = retail_sales_df, name = "retail_sales", freq = "D"),
        # Dataset(df = yosemite_temps_df, name = "yosemite_temps", freq = "5min"),
        # Dataset(df = ercot_load_df, name = "ercot_load", freq = "H"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"epochs": EPOCHS, "seasonality_mode": "multiplicative", "learning_rate": 0.1}),
        (NeuralProphetModel, {"epochs": EPOCHS, "learning_rate": 0.1}),
        # (ProphetModel, {}),
    ]
    log.info("SimpleBenchmark")
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_train, results_test = benchmark.run()
    log.debug(results_test.to_string())
    log.info("#### Done with test_simple_benchmark")


def test_cv_benchmark():
    log.info("test_cv_benchmark")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
        # Dataset(df = retail_sales_df, name = "retail_sales", freq = "D"),
        # Dataset(df = yosemite_temps_df, name = "yosemite_temps", freq = "5min"),
        # Dataset(df = ercot_load_df, name = "ercot_load", freq = "H"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"epochs": EPOCHS, "learning_rate": 0.1, "seasonality_mode": "multiplicative"}),
        # (NeuralProphetModel, {"epochs": EPOCHS, "learning_rate": 0.1}),
        # (ProphetModel, {}),
        # (ProphetModel, {"seasonality_mode": "multiplicative"}),
    ]

    benchmark_cv = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=10,
        num_folds=3,
        fold_overlap_pct=0,
        save_dir=SAVE_DIR,
        num_processes=1,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug(results_summary.to_string())
    log.debug(results_train.to_string())
    log.debug(results_test.to_string())
    log.info("#### Done with test_cv_benchmark")
