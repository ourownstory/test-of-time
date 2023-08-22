import logging
import os
import pathlib

import pandas as pd
import pytest

from darts.models import NaiveDrift

from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.lgbm import LightGBMModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.random_forest import RandomForest

from tot.benchmark import SimpleBenchmark
from tot.datasets.dataset import Dataset
from tot.evaluation.metrics import ERROR_FUNCTIONS
from tot.models.models_darts import DartsForecastingModel, LinearRegressionModel, RandomForestModel
from tot.models.models_naive import NaiveModel, SeasonalNaiveModel
from tot.models.models_neuralprophet import NeuralProphetModel, TorchProphetModel
from tot.models.models_prophet import ProphetModel
from tot.evaluation.metric_utils import _calc_metrics_for_single_ID_and_every_fcst_step, calculate_metrics_by_ID_for_forecast_step

RNN = {
    "Dmodel": RNNModel,
    "model": "LSTM",
    "input_chunk_length": 3*24, # Increased to capture more temporal information
    "output_chunk_length": 1,
    "hidden_dim": 16, # Increased for greater model complexity
    "n_rnn_layers": 3, # Increased to add complexity
    "batch_size": 128, # Increased for faster computation
    "n_epochs": 30, # Increased for better training
    "random_state": 0,
    "training_length": 5*24, # Increased to capture more temporal information
    "force_reset": True,
    "n_lags": 3*24, # Increased for better performance
    "n_forecasts": 33,
    "pl_trainer_kwargs": {"accelerator": "cpu", "devices": 1},
    "_data_params": {},
    "optimizer_kwargs": {"lr": 1e-4},
    "dropout": 0.1,
}

LGBM = {
    "Dmodel": LightGBMModel,
    "n_forecasts": 1,
    "output_chunk_length": 1,
    "n_lags": 4, # Increased for better performance
    "lags": 4, # Aligned with n_lags
    #"_data_params": {},
}

TorchProphet = {
    "yearly_seasonality": 14, # Increased to capture more seasonality
    "weekly_seasonality": 6, # Adjusted to the number of days in a week
    "daily_seasonality": 0, # Enabled daily seasonality
    "epochs": 30, # Increased for better training
    "batch_size": 128, # Increased for faster computation
    "season_global_local": "local",
    "trend_global_local": "local",
    "n_changepoints": 40, # Reduced to prevent overfitting
    "learning_rate": 0.0009195786690458936, #Slightly increased for faster convergence
    "newer_samples_weight": 2.0,
}

NBEATS = {
    "Dmodel": NBEATSModel,
    "input_chunk_length": 3*24, # Increased to capture more temporal information
    "output_chunk_length": 33,
    "generic_architecture": True,
    "n_lags": 3*24, # Increased for better performance
    "n_forecasts": 33,
    "n_epochs": 30, # Increased for better training
    "batch_size": 128, # Increased for faster computation
    "num_layers": 4, # Reduced to prevent overfitting
    "num_stacks": 8, # Reduced to prevent overfitting
    "layer_widths": 256, # Reduced to prevent overfitting
}

TCN = {
    "Dmodel": TCNModel,
    "input_chunk_length": 3*24, # Increased to capture more temporal information
    "output_chunk_length": 33,
    "n_epochs": 30, # Increased for better training
    "batch_size": 128, # Increased for faster computation
    "dropout": 0.2, # Increased for regularization
    "dilation_base": 2,
    "weight_norm": True,
    "kernel_size": 4, # Increased for larger receptive field
    "num_filters": 8, # Increased for capturing more features
    "n_forecasts": 33,
    "n_lags": 3*24, # Increased for better performance
}

Regression = {
    "n_forecasts": 33,
    "output_chunk_length": 33,
    "lags": 3*24, # Increased for better performance
}