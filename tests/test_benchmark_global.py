#!/usr/bin/env python3

import pytest
import os
import pathlib
import pandas as pd
import logging
import matplotlib.pyplot as plt

from tot.dataset import Dataset
from tot.models import NeuralProphetModel, ProphetModel
from tot.experiment import SimpleExperiment, CrossValidationExperiment
from tot.benchmark import SimpleBenchmark, CrossValidationBenchmark
from tot.benchmark import ManualBenchmark, ManualCVBenchmark
from tot.metrics import ERROR_FUNCTIONS

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

NROWS = 200
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0
ERCOT_REGIONS = ["NORTH", "EAST", "FAR_WEST"]

PLOT = False



def test_benchmark_simple_global_modeling():
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
        (
            NeuralProphetModel,
            {
                "n_lags": 24,
                "n_forecasts": 8,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": False,
                "global_time_normalization": False,
            },
        ),
        # (NeuralProphetModel, {"n_lags": 24, "n_forecasts": 8, "learning_rate": 0.1, "epochs": EPOCHS, "global_normalization": True, "global_time_normalization": False, "unknown_data_normalization": False}),
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


def test_benchmark_CV_global_modeling():
    ercot_df_aux = pd.read_csv(ERCOT_FILE)
    ercot_df = pd.DataFrame()
    for region in ERCOT_REGIONS:
        ercot_df = pd.concat(
            (ercot_df, ercot_df_aux[ercot_df_aux["ID"] == region].iloc[:NROWS].copy(deep=True)), ignore_index=True
        )
    peyton_manning_df_aux = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    peyton_manning_df = pd.DataFrame()
    slice_idx = 0
    for df_name in ["df1", "df2"]:
        df_aux = peyton_manning_df_aux.iloc[slice_idx : slice_idx + 100]
        df_aux = df_aux.assign(ID=df_name)
        peyton_manning_df = pd.concat((peyton_manning_df, df_aux), ignore_index=True)
        slice_idx = slice_idx + 100

    dataset_list = [
        Dataset(df=ercot_df, name="ercot_load", freq="H"),
        Dataset(df=peyton_manning_df, name="peyton_manning_many_ts", freq="D"),
    ]

    model_classes_and_params = [
        (
            NeuralProphetModel,
            {
                "n_lags": 10,
                "n_forecasts": 5,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": False,
                "global_time_normalization": True,
            },
        ),
        (
            NeuralProphetModel,
            {
                "n_lags": 10,
                "n_forecasts": 5,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": True,
                "global_time_normalization": True,
            },
        ),
        (
            NeuralProphetModel,
            {
                "n_lags": 10,
                "n_forecasts": 5,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": False,
                "global_time_normalization": False,
            },
        ),
        # (NeuralProphetModel, {"n_lags": 24, "n_forecasts": 8, "learning_rate": 0.1, "epochs": EPOCHS, "global_normalization": True, "global_time_normalization": False, "unknown_data_normalization": False}),
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
        ercot = results_summary[results_summary["split"] == "test"]
        plt_ercot = ercot.plot(x="data", y="MASE", kind="barh")
        plt.show()


def test_benchmark_manual_global_modeling():
    ercot_df_aux = pd.read_csv(ERCOT_FILE)
    ercot_df = pd.DataFrame()
    for region in ERCOT_REGIONS:
        ercot_df = pd.concat(
            (ercot_df, ercot_df_aux[ercot_df_aux["ID"] == region].iloc[:NROWS].copy(deep=True)), ignore_index=True
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
            params={"n_lags": 5, "n_forecasts": 3, "epochs": EPOCHS, "learning_rate": 0.1},
            data=Dataset(df=ercot_df, name="ercot_load", freq="H"),
            metrics=metrics,
            test_percentage=25,
        ),
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1, "epochs": EPOCHS},
            data=Dataset(df=peyton_manning_df, name="peyton_manning_many_ts", freq="D"),
            metrics=metrics,
            test_percentage=25,
        ),
    ]
    benchmark = ManualBenchmark(
        experiments=experiments,
        metrics=metrics,
        save_dir=SAVE_DIR,
    )
    results_train, results_test = benchmark.run()
    log.debug("{}".format(results_test))


def test_benchmark_manualCV_global_modeling():
    ercot_df_aux = pd.read_csv(ERCOT_FILE)
    ercot_df = pd.DataFrame()
    for region in ERCOT_REGIONS:
        ercot_df = pd.concat(
            (ercot_df, ercot_df_aux[ercot_df_aux["ID"] == region].iloc[:NROWS].copy(deep=True)), ignore_index=True
        )
    peyton_manning_df_aux = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    peyton_manning_df = pd.DataFrame()
    slice_idx = 0
    log.info("Creating a date intersection between df1 and df2")
    for df_name in ["df1", "df2"]:
        df_aux = peyton_manning_df_aux.iloc[slice_idx : slice_idx + 100]
        df_aux = df_aux.assign(ID=df_name)
        peyton_manning_df = pd.concat((peyton_manning_df, df_aux), ignore_index=True)
        slice_idx = slice_idx + 100
    log.info("Creating an intersection between df1 and df2.")
    overlap_dates = pd.Series(pd.date_range(start="2008-02-15", end="2008-03-24", freq="D"))
    overlap_vals = pd.Series(range(len(overlap_dates)))
    peyton_manning_df_intersect = pd.DataFrame()
    peyton_manning_df_intersect["ds"] = overlap_dates
    peyton_manning_df_intersect["y"] = overlap_vals
    peyton_manning_df_intersect["ID"] = "df2"
    peyton_manning_df_intersect = pd.concat((peyton_manning_df.iloc[:101],peyton_manning_df_intersect, peyton_manning_df.iloc[101:]), ignore_index=True)
    peyton_manning_df_intersect["ds"] = pd.to_datetime(peyton_manning_df_intersect["ds"])

    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"n_lags": 5, "n_forecasts": 3, "epochs": EPOCHS, "learning_rate": 0.1},
            data=Dataset(df=ercot_df, name="ercot_load", freq="H"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="local",
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS, "seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=ercot_df, name="ercot_load", freq="H"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="intersect",
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"n_lags": 5, "n_forecasts": 3, "epochs": EPOCHS, "learning_rate": 0.1},
            data=Dataset(df=peyton_manning_df, name="peyton_manning_many_ts", freq="D"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="local",
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS, "seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=peyton_manning_df_intersect, name="peyton_manning_many_ts", freq="D"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="intersect",
        ),
    ]
    benchmark_cv = ManualCVBenchmark(
        experiments=experiments,
        metrics=metrics,
        save_dir=SAVE_DIR,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug("{}".format(results_summary))


def test_benchmark_dict_global_modeling():
    # It will be deprecated soon
    ercot_df_aux = pd.read_csv(ERCOT_FILE)
    ercot_dict = {}
    for region in ERCOT_REGIONS:
        aux = ercot_df_aux[ercot_df_aux["ID"] == region].iloc[:NROWS].copy(deep=True)
        aux.drop("ID", axis=1, inplace=True)
        ercot_dict[region] = aux
    dataset_list = [
        Dataset(df=ercot_dict, name="ercot_load", freq="H"),
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
        # (NeuralProphetModel, {"n_lags": 24, "n_forecasts": 8, "learning_rate": 0.1, "epochs": EPOCHS, "global_normalization": True, "global_time_normalization": False, "unknown_data_normalization": False}),
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

