# !/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

from tot.benchmark import CrossValidationBenchmark, ManualCVBenchmark, SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.experiment import CrossValidationExperiment
from tot.models.models_naive import NaiveModel, SeasonalNaiveModel
from tot.models.models_neuralprophet import NeuralProphetModel, TorchProphetModel
from tot.models.models_simple import LinearRegressionModel

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


def test_benchmark_panel_data_input():
    # test all relevant models on their default config for simple panel dataset
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
            TorchProphetModel,
            {
                "learning_rate": 0.1,
                "epochs": EPOCHS,
                "global_normalization": True,
                "global_time_normalization": True,
            },
        ),
        (LinearRegressionModel, {"lags": 24, "output_chunk_length": 8, "n_forecasts": 8}),
        (NaiveModel, {"n_forecasts": 8}),
        (SeasonalNaiveModel, {"n_forecasts": 8, "season_length": 24}),
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


def test_benchmark_CV_panel_data_input():
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
                "n_lags": 10,
                "n_forecasts": 5,
                "learning_rate": 0.1,
                "epochs": EPOCHS,
            },
        ),
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark_cv_1 = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MASE", "RMSE"],
        test_percentage=0.1,
        num_folds=3,
        fold_overlap_pct=0,
    )
    results_summary_1, results_train_1, results_test_1 = benchmark_cv_1.run()
    log.debug("{}".format(results_summary_1))

    benchmark_cv_2 = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MASE", "RMSE"],
        test_percentage=0.1,
        num_folds=3,
        fold_overlap_pct=0.2,
    )
    results_summary_2, results_train_2, results_test_2 = benchmark_cv_2.run()
    log.debug("{}".format(results_summary_2))
    if PLOT:
        results_summary_2[results_summary_2["split"] == "test"]


def test_benchmark_manualCV_global_modeling():
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
    peyton_manning_df_intersect = pd.concat(
        (
            peyton_manning_df.iloc[:100],
            peyton_manning_df_intersect,
            peyton_manning_df.iloc[101:],
        ),
        ignore_index=True,
    )
    peyton_manning_df_intersect["ds"] = pd.to_datetime(peyton_manning_df_intersect["ds"])

    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        # test local split on dataset of equal length
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={
                "n_lags": 5,
                "n_forecasts": 3,
                "epochs": EPOCHS,
                "learning_rate": 0.1,
            },
            data=Dataset(df=ercot_df, name="ercot_load", freq="H"),
            metrics=metrics,
            test_percentage=0.1,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="local",
        ),
        # test global-time split on dataset of equal length
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={
                "epochs": EPOCHS,
                "seasonality_mode": "multiplicative",
                "learning_rate": 0.1,
            },
            data=Dataset(df=ercot_df, name="ercot_load", freq="H"),
            metrics=metrics,
            test_percentage=0.1,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="global-time",
        ),
        # test local split on dataset with ts of different start and end dates
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={
                "n_lags": 5,
                "n_forecasts": 3,
                "epochs": EPOCHS,
                "learning_rate": 0.1,
            },
            data=Dataset(df=peyton_manning_df, name="peyton_manning_many_ts", freq="D"),
            metrics=metrics,
            test_percentage=0.1,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="local",
        ),
        # test intersect split on dataset with ts of unequal length
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={
                "epochs": EPOCHS,
                "seasonality_mode": "multiplicative",
                "learning_rate": 0.1,
            },
            data=Dataset(
                df=peyton_manning_df_intersect,
                name="peyton_manning_many_ts",
                freq="D",
            ),
            metrics=metrics,
            test_percentage=0.1,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="intersect",
        ),
        # test global-time split on dataset with ts of unequal length
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={
                "epochs": EPOCHS,
                "seasonality_mode": "multiplicative",
                "learning_rate": 0.1,
            },
            data=Dataset(
                df=peyton_manning_df_intersect,
                name="peyton_manning_many_ts",
                freq="D",
            ),
            metrics=metrics,
            test_percentage=0.1,
            num_folds=3,
            fold_overlap_pct=0,
            global_model_cv_type="global-time",
        ),
    ]
    benchmark_cv = ManualCVBenchmark(
        experiments=experiments,
        metrics=metrics,
        save_dir=SAVE_DIR,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug("{}".format(results_summary))
