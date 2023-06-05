import multiprocessing
import time
import pandas as pd
from experiments.pipeline_experiment import run
from experiments.utils import generate_one_shape_season_and_ar_data
from tot.models.models_darts import DartsForecastingModel
from darts.models.forecasting.lgbm import LightGBMModel

def run_benchmark():
    start_time=time.time()
    PLOT = False
    DIR_NAME = "SEASON_balanced_amplitude_scale_ar_LGBM"
    FREQ = "H"
    SERIES_LENGTH = 24 * 7 * 15
    DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
    MODEL = DartsForecastingModel
    MODEL_PARAMS = {
        "model": LightGBMModel,
        "n_forecasts": 1,
        "output_chunk_length": 1,
        "lags": 4,
        "n_lags": 4,
        "multi_models": False,
        "_data_params": {},
    }

    df = generate_one_shape_season_and_ar_data(
        series_length=SERIES_LENGTH,
        date_rng=DATE_RNG,
        n_ts_groups=[5, 5],
        offset_per_group=[100, 1000],
        amplitude_per_group=[5, 50],
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

