import pandas as pd

from experiments.pipeline_experiment import run
from experiments.utils import generate_canceling_shape_season_and_ar_data
from tot.models import NeuralProphetModel

PLOT = False
DIR_NAME = "SEA_SHAPE_unbalanced_amplitude_twloam_1_2_AR_NP"
FREQ = "H"
SERIES_LENGTH = 24 * 7 * 15
DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
MODEL = NeuralProphetModel
MODEL_PARAMS = {
    "n_forecasts": 1,
    "epochs": 30,
    "global_normalization": True,
    "normalize": "off",
    "n_lags": 4,
}

df = generate_canceling_shape_season_and_ar_data(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[1, 2],
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
    metrics=["MAE", "RMSE", "MAPE", "MASE"],
    test_percentage=0.25,
    plot=PLOT,
)
