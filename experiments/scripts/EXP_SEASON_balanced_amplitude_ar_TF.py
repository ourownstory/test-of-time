import multiprocessing
import time
import pandas as pd
from experiments.pipeline_experiment import run
from experiments.utils import generate_one_shape_season_and_ar_data
from tot.models.models_darts import DartsForecastingModel
from darts.models.forecasting.transformer_model import TransformerModel

def run_benchmark():
    start_time=time.time()
    PLOT = False
    DIR_NAME = "SEASON_balanced_amplitude_NP"
    FREQ = "H"
    SERIES_LENGTH = 24 * 7 * 15
    DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
    MODEL = DartsForecastingModel
    MODEL_PARAMS = {
        "model": TransformerModel,
        'model_name': 'air_transformer',
        "n_forecasts": 1,
        'n_lags': 4,
        "output_chunk_length": 1,
        "input_chunk_length": 4,
        'batch_size': 128,
        'n_epochs': 100,
        'nr_epochs_val_period': 10,
        'd_model': 16,
        # 'n_heads':8,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'activation': 'relu',
        'random_state': 42,
        'save_checkpoints': True,
        'force_reset': True,
        "_data_params": {},
    }

    df = generate_one_shape_season_and_ar_data(
        series_length=SERIES_LENGTH,
        date_rng=DATE_RNG,
        n_ts_groups=[5, 5],
        offset_per_group=[0, 0],
        amplitude_per_group=[50, 5],
    )

    run(
        dir_name=DIR_NAME,
        save=True,
        df=df,
        df_name=DIR_NAME,
        freq=FREQ,
        model_class=MODEL,
        model_params=MODEL_PARAMS,
        scalers="default",
        scaling_levels="default",
        reweight_loss=True,
        metrics=["MAE", "RMSE", "MASE"],
        test_percentage=0.25,
        plot=PLOT,
    )
    end_time=time.time()
    print("time taken",end_time-start_time)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_benchmark()

