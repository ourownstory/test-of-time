#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

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


def test_simple_experiment():
    log.info("test_simple_experiment")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {
        "seasonality_mode": "multiplicative",
        "epochs": EPOCHS,
        "learning_rate": 0.1,
    }
    exp = SimpleExperiment(
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
    )
    result_train, result_val = exp.run()
    log.debug(result_val)
    log.info("#### Done with test_simple_experiment")


def test_cv_experiment():
    log.info("test_cv_experiment")
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {"epochs": EPOCHS, "learning_rate": 0.1}
    log.info("CrossValidationExperiment")
    exp_cv = CrossValidationExperiment(
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=10,
        num_folds=2,
        fold_overlap_pct=0,
        save_dir=SAVE_DIR,
    )
    result_train, result_val = exp_cv.run()
    log.debug(result_val)
    log.info("#### Done with test_cv_experiment")
