import pandas as pd

from experiments.pipeline_experiment import run
from experiments.utils import generate_one_shape_season_and_ar_and_trend_data
from tot.models import NeuralProphetModel

PLOT = False
DIR_NAME = "TREND_SEASON_balanced_amplitude_NP"
FREQ = "H"
SERIES_LENGTH = 24 * 2 * 15
DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
MODEL = NeuralProphetModel
MODEL_PARAMS = {
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "n_lags": 4,
}

df = generate_one_shape_season_and_ar_and_trend_data(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[5, 5],
    offset_per_group=[0, 0],
    trend_gradient_per_group=[1, .1],
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
    metrics=["MAE", "RMSE", "MAPE", "MASE"],
    test_percentage=0.25,
    plot=PLOT,
)
