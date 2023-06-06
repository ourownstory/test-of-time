import multiprocessing
import time
import pandas as pd
from experiments.pipeline_experiment import run
from experiments.utils import generate_intermittent
from tot.models import NeuralProphetModel

def run_benchmark():
    start_time=time.time()
    PLOT = False
    DIR_NAME = "SEASON_balanced_amplitude_scale_ar_outlier_NP_localST"
    FREQ = "H"
    SERIES_LENGTH = 24 * 7 * 15
    DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
    MODEL = NeuralProphetModel
    MODEL_PARAMS = {
        "n_forecasts": 1,
        "epochs": 30,
        "global_normalization": True,
        "normalize": "off",
        "trend_global_local": "local",
        "season_global_local": "local",
        "n_lags": 4,
    }

    df = generate_intermittent(
        series_length=SERIES_LENGTH,
        date_rng=DATE_RNG,
        n_ts_groups=[5, 5],
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
