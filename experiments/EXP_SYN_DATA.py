import argparse
import multiprocessing
import time

import pandas as pd
from darts.models.forecasting.lgbm import LightGBMModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.transformer_model import TransformerModel

from experiments.pipeline_experiment import run
from experiments.utils import (
    gen_cancel_shape_ar,
    gen_cancel_shape_ar_outlier_0p1,
    gen_cancel_shape_ar_outlier_1p,
    gen_model_and_params,
    gen_model_and_params_none,
    gen_model_and_params_norm,
    gen_one_shape_ar,
    gen_one_shape_ar_outlier_0p1,
    gen_one_shape_ar_outlier_1p,
    gen_one_shape_ar_trend,
    gen_one_shape_ar_trend_cp,
    gen_one_shape_heteroscedacity,
    gen_one_shape_heteroscedacity_op,
    gen_struc_break_mean,
    gen_struc_break_var,
    generate_intermittent,
)
from tot.models import NaiveModel, NeuralProphetModel, SeasonalNaiveModel, TorchProphetModel
from tot.models.models_darts import DartsForecastingModel

FUNCTIONS = {
    "gen_one_shape_ar": gen_one_shape_ar,
    "gen_one_shape_ar_outlier_0p1": gen_one_shape_ar_outlier_0p1,
    "gen_one_shape_ar_outlier_1p": gen_one_shape_ar_outlier_1p,
    "gen_cancel_shape_ar": gen_cancel_shape_ar,
    "gen_cancel_shape_ar_outlier_0p1": gen_cancel_shape_ar_outlier_0p1,
    "gen_cancel_shape_ar_outlier_1p": gen_cancel_shape_ar_outlier_1p,
    "gen_one_shape_ar_trend": gen_one_shape_ar_trend,
    "gen_one_shape_ar_trend_cp": gen_one_shape_ar_trend_cp,
    "generate_intermittent": generate_intermittent,
    "gen_one_shape_heteroscedacity": gen_one_shape_heteroscedacity,
    "gen_one_shape_heteroscedacity_op": gen_one_shape_heteroscedacity_op,
    "gen_struc_break_mean": gen_struc_break_mean,
    "gen_struc_break_var": gen_struc_break_var,
}

PARAMS = {
    "NP": {
        "n_forecasts": 1,
        "epochs": 30,
        "global_normalization": True,
        "normalize": "off",
        "n_lags": 4,
    },
    "NP_localST": {
        "n_forecasts": 1,
        "epochs": 30,
        "global_normalization": True,
        "normalize": "off",
        "trend_global_local": "local",
        "season_global_local": "local",
        "n_lags": 4,
    },
    "NP_FNN": {
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
    },
    "NP_FNN_wb": {
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
    },
    "TP": {
        "n_forecasts": 1,
        "epochs": 30,
        "global_normalization": True,
        "normalize": "off",
    },
    "TP_localST": {
        "n_forecasts": 1,
        "epochs": 30,
        "global_normalization": True,
        "normalize": "off",
        "trend_global_local": "local",
        "season_global_local": "local",
    },
    "LGBM": {
        "model": LightGBMModel,
        "n_forecasts": 1,
        "output_chunk_length": 1,
        "lags": 4,
        "n_lags": 4,
        "_data_params": {},
    },
    "RNN": {
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
    },
    "RNN_wb": {
        "model": RNNModel,
        "input_chunk_length": 4,
        "hidden_dim": 16,
        "n_rnn_layers": 1,
        "batch_size": 128,
        "n_epochs": 30,
        "random_state": 0,
        "training_length": 4,
        "force_reset": True,
        "n_lags": 4,
        "n_forecasts": 1,
        "pl_trainer_kwargs": {"accelerator": "gpu", "devices": 1},
        "_data_params": {},
    },
    "TF": {
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
    },
    "Naive": {"n_forecasts": 1},
    "SNaive": {"n_forecasts": 1, "season_length": 24},
}

MODELS = {
    "NeuralProphetModel": NeuralProphetModel,
    "TorchProphetModel": TorchProphetModel,
    "LightGBMModel": DartsForecastingModel,
    "RNNModel": DartsForecastingModel,
    "TransformerModel": DartsForecastingModel,
    "NaiveModel": NaiveModel,
    "SeasonalNaiveModel": SeasonalNaiveModel,
}
GEN_FUNC = {
    "gen_model_and_params": gen_model_and_params,
    "gen_model_and_params_norm": gen_model_and_params_norm,
    "gen_model_and_params_none": gen_model_and_params_none,
}


def run_benchmark(
    model,
    params,
    data_func,
    n_ts_groups,
    amplitude_per_group,
    gen_func,
    offset_per_group=[0, 0],
    data_trend_gradient_per_group=None,
    proportion_break=None,
):
    start_time = time.time()
    PLOT = False
    FREQ = "H"
    SERIES_LENGTH = 24 * 7 * 15
    DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")

    # The data_func, model, and params arguments are now provided as arguments
    MODEL = MODELS[model]
    MODEL_PARAMS = PARAMS[params]
    DIR_NAME = "{}_{}_n_ts_{}_am_{}_of_{}_gr_{}".format(
        data_func, params, n_ts_groups, amplitude_per_group, offset_per_group, data_trend_gradient_per_group
    )
    if params == "TF" or params == "RNN" or params == "RNN_wb":
        NUM_PROCESSES = 1
    else:
        NUM_PROCESSES = 19

    if data_trend_gradient_per_group is not None:
        df = FUNCTIONS[data_func](
            series_length=SERIES_LENGTH,
            date_rng=DATE_RNG,
            n_ts_groups=n_ts_groups,
            offset_per_group=offset_per_group,
            amplitude_per_group=amplitude_per_group,
            trend_gradient_per_group=data_trend_gradient_per_group,
        )
    elif proportion_break is not None:
        df = FUNCTIONS[data_func](
            series_length=SERIES_LENGTH,
            date_rng=DATE_RNG,
            n_ts_groups=n_ts_groups,
            offset_per_group=offset_per_group,
            amplitude_per_group=amplitude_per_group,
            proportion_break=proportion_break,
        )
    else:
        df = FUNCTIONS[data_func](
            series_length=SERIES_LENGTH,
            date_rng=DATE_RNG,
            n_ts_groups=n_ts_groups,
            offset_per_group=offset_per_group,
            amplitude_per_group=amplitude_per_group,
        )

    run(
        dir_name=DIR_NAME,
        save=True,
        df=df,
        df_name="",
        freq=FREQ,
        model_class=MODEL,
        model_params=MODEL_PARAMS,
        scalers="default",
        scaling_levels="default",
        reweight_loss=True,
        metrics=["MAE", "RMSE", "MASE"],
        test_percentage=0.25,
        plot=PLOT,
        num_processes=NUM_PROCESSES,
        model_and_params_generator=GEN_FUNC[gen_func],
    )
    end_time = time.time()
    print("time taken", end_time - start_time)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run a benchmark")
    parser.add_argument("--data_func", type=str, required=True, help="Data function")
    parser.add_argument(
        "--data_n_ts_groups",
        type=str,
        required=True,
        default="1,1",
        help="Number of timer series per group in data function",
    )
    parser.add_argument(
        "--data_offset_per_group", type=str, required=True, default="0,0", help="Offsets per group in data function"
    )
    parser.add_argument(
        "--data_amplitude_per_group",
        type=str,
        required=True,
        default="0,0",
        help="Amplitudes per group in data function",
    )
    parser.add_argument(
        "--data_trend_gradient_per_group",
        type=str,
        required=False,
        default=None,
        help="Optional argument - Trend gradient per group in data ",
    )
    parser.add_argument(
        "--proportion_break",
        type=str,
        required=False,
        default=None,
        help="Optional argument - Proportion of breaks in data function",
    )
    parser.add_argument("--model", type=str, required=True, help="Model class")
    parser.add_argument("--params", type=str, required=True, help="Model parameters")
    parser.add_argument(
        "--gen_func", type=str, required=False, default="gen_model_and_params", help="Param generation function"
    )

    args = parser.parse_args()

    # Freezing support for multiprocessing
    multiprocessing.freeze_support()

    # post-processing args
    args.data_n_ts_groups = [int(i) for i in args.data_n_ts_groups.split(",")]
    args.data_offset_per_group = [int(i) for i in args.data_offset_per_group.split(",")]
    args.data_amplitude_per_group = [int(i) for i in args.data_amplitude_per_group.split(",")]
    args.data_trend_gradient_per_group = (
        [float(i) for i in args.data_trend_gradient_per_group.split(",")]
        if args.data_trend_gradient_per_group is not None
        else None
    )
    args.proportion_break = (
        [int(i) for i in args.proportion_break.split(",")] if args.proportion_break is not None else None
    )

    # Running benchmark
    run_benchmark(
        model=args.model,
        params=args.params,
        data_func=args.data_func,
        n_ts_groups=args.data_n_ts_groups,
        offset_per_group=args.data_offset_per_group,
        amplitude_per_group=args.data_amplitude_per_group,
        data_trend_gradient_per_group=args.data_trend_gradient_per_group,
        gen_func=args.gen_func,
        proportion_break=args.proportion_break,
    )
