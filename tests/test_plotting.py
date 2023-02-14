#!/usr/bin/env python3
import logging
import os
import pathlib

import pandas as pd
import pytest

from tot.benchmark import SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.evaluation.metrics import ERROR_FUNCTIONS
from tot.models.models_naive import NaiveModel, SeasonalNaiveModel
from tot.models.models_neuralprophet import NeuralProphetModel
from tot.models.models_simple import LinearRegressionModel, ProphetModel
from tot.plot import plot

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

# plot tests cover both plotting backends
decorator_input = ["plotting_backend", ["plotly-auto", "plotly", "plotly-resampler"]]


@pytest.mark.parametrize(*decorator_input)
def test_basic_plot(plotting_backend):
    log.info("test_basic_plot")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"n_forecasts": 4, "n_lags": 3, "epochs": 1}),
        (NaiveModel, {"n_forecasts": 4}),
        (SeasonalNaiveModel, {"n_forecasts": 4, "season_length": 12}),
        (ProphetModel, {}),
        (LinearRegressionModel, {"lags": 12, "output_chunk_length": 1, "n_forecasts": 4}),
    ]

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=0.25,
        save_dir=SAVE_DIR,
        num_processes=1,
    )

    results_train, results_test = benchmark.run()
    log.debug(results_test.to_string())
    fig = plot(benchmark.fcst_test[0], plotting_backend=plotting_backend)
    if PLOT:
        fig.show()
    log.info("#### Done with test_basic_plot")
