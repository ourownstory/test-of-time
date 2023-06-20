import logging
from functools import reduce

from darts.models.forecasting.lgbm import LightGBMModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.transformer_model import TransformerModel

__all__ = ["get_num_processes", "get_params_for_model", "SUPPORTED_PARAMS"]

log = logging.getLogger("experiments")


def get_num_processes(params_name):
    if params_name in ["TF", "RNN", "RNN_wb", "LGBM", "NP_FNN_sw_wb"]:
        return 1
    return 10


def get_params_for_model(model_name, params_name):
    try:
        return PARAMS[model_name][params_name]
    except KeyError as e:
        log.error(e, f"Invalid pre-defined params name for {model_name}. Available params: {PARAMS[model_name].keys()}")


# NeuralProphetModel pre-defined params
NP = {  # NeuralProphetModel with global trend, global seasonality and autoregression enabled.
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "n_lags": 4,
}
NP_localST = {  # NeuralProphetModel with local trend, local seasonality and autoregression enabled.
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "trend_global_local": "local",
    "season_global_local": "local",
    "n_lags": 4,
}
NP_FNN = {  # NeuralProphetModel with only autoregression enabled.
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "n_changepoints": 0,
    "growth": "off",
    "n_lags": 4,
}
NP_FNN_wb = {  # NeuralProphetModel with only autoregression enabled used for window-based normalization.
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "n_changepoints": 0,
    "growth": "off",
    "n_lags": 24,
}
NP_FNN_sw_wb = (
    {  # NeuralProphetModel with only autoregression with 1 hidden layer enabled used for window-based normalization.
        "n_forecasts": 1,
        "epochs": 30,
        "global_normalization": True,
        "normalize": "off",
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "n_changepoints": 0,
        "growth": "off",
        "n_lags": 24,
        "ar_layers": [128],
    }
)

# TorchProphetModel pre-defined params
TP = {  # TorchProphetModel with global trend and global seasonality enabled.
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
}
TP_localST = {  # TorchProphetModel with local trend and local seasonality enabled.
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "trend_global_local": "local",
    "season_global_local": "local",
}

# LightGBMModel pre-defined params
LGBM = {
    "model": LightGBMModel,
    "n_forecasts": 1,
    "output_chunk_length": 1,
    "lags": 4,
    "n_lags": 4,
    "_data_params": {},
}

# RNN pre-defined params

RNN = {
    "model": RNNModel,
    "input_chunk_length": 4,
    "hidden_dim": 16,
    "n_rnn_layers": 1,
    "batch_size": 128,
    "n_epochs": 80,
    "random_state": 0,
    "training_length": 4,
    "force_reset": True,
    "n_lags": 4,
    "n_forecasts": 1,
    "pl_trainer_kwargs": {"accelerator": "gpu", "devices": 1},
    "_data_params": {},
}

RNN_wb_in = {  # RNN params used for window-based instance normalization.
    "model": RNNModel,
    "input_chunk_length": 24,
    "hidden_dim": 16,
    "n_rnn_layers": 1,
    "batch_size": 128,
    "n_epochs": 30,
    "random_state": 0,
    "training_length": 24,
    "force_reset": True,
    "n_lags": 24,
    "n_forecasts": 1,
    "_data_params": {},
}
RNN_wb_ba = {  # RNN params used for window-based batch normalization.
    "model": RNNModel,
    "input_chunk_length": 4,
    "hidden_dim": 16,
    "n_rnn_layers": 1,
    "batch_size": 128,
    "n_epochs": 30,
    "random_state": 0,
    "training_length": 24,
    "force_reset": True,
    "n_lags": 24,
    "n_forecasts": 1,
    "pl_trainer_kwargs": {"accelerator": "gpu", "devices": 1},
    "_data_params": {},
}

# TransformerModel pre-defined params
TF = {
    "model": TransformerModel,
    "model_name": "air_transformer",
    "n_forecasts": 1,
    "n_lags": 4,
    "output_chunk_length": 1,
    "input_chunk_length": 4,
    "batch_size": 128,
    "n_epochs": 100,
    "nr_epochs_val_period": 10,
    "d_model": 16,
    # 'n_heads':8,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "activation": "relu",
    "random_state": 42,
    "save_checkpoints": True,
    "force_reset": True,
    "pl_trainer_kwargs": {"accelerator": "gpu", "devices": 1},
    "_data_params": {},
}

# Naive pre-defined params
Naive = ({"n_forecasts": 1},)
SNaive = ({"n_forecasts": 1, "season_length": 24},)


#  TODO: Maybe set more meaningful names for pre-defined params

PARAMS = {
    "NeuralProphetModel": {
        "NP": NP,
        "NP_localST": NP_localST,
        "NP_FNN": NP_FNN,
        "NP_FNN_wb": NP_FNN_wb,
        "NP_FNN_sw_wb": NP_FNN_sw_wb,
    },
    "TorchProphetModel": {"TP": TP, "TP_localST": TP_localST},
    "LightGBMModel": {"LGBM": LGBM},
    "RNNModel": {
        "RNN": RNN,
        "RNN_wb": RNN_wb_in,
        "RNN_wb_nl": RNN_wb_ba,
    },
    "TransformerModel": {"TF": TF},
    "NaiveModel": {"Naive": Naive},
    "SeasonalNaiveModel": {"SNaive": SNaive},
}

SUPPORTED_PARAMS = reduce(lambda l1, l2: l1 + l2, [list(model.keys()) for model in PARAMS.values()])
