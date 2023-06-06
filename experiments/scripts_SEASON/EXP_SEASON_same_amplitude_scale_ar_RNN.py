import multiprocessing
import time
import pandas as pd
from experiments.pipeline_experiment import run
from experiments.utils import generate_one_shape_season_and_ar_data
from darts.models.forecasting.rnn_model import RNNModel
from tot.models.models_darts import DartsForecastingModel

def run_benchmark():
    start_time=time.time()
    PLOT = False
    DIR_NAME = "SEASON_same_amplitude_scale_ar_RNN"
    FREQ = "H"
    SERIES_LENGTH = 24 * 7 * 15
    DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
    MODEL = DartsForecastingModel
    MODEL_PARAMS = {
        "model": RNNModel,
        "input_chunk_length": 4,
        'hidden_dim': 16,
        'n_rnn_layers': 1,
        'batch_size': 128,
        'n_epochs': 30,
        'random_state': 0,
        'training_length': 4,
        'force_reset': True,
        'n_lags': 4,
        'n_forecasts': 1,
        '_data_params': {},
    }


    df = generate_one_shape_season_and_ar_data(
        series_length=SERIES_LENGTH,
        date_rng=DATE_RNG,
        n_ts_groups=[5, 5],
        offset_per_group=[100, 100],
        amplitude_per_group=[50, 50],
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

