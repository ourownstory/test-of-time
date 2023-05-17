from tot.models import NeuralProphetModel
from tot.normalization.experiments.pipeline import generate_canceling_shape_season_data
from tot.normalization.experiments_reweighting.pipeline_reweighting import run
import pandas as pd

SERIES_LENGTH = 24 * 7 * 15
DATE_RNG = date_rng = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")

df = generate_canceling_shape_season_data(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[2, 1],
    offset_per_group=[0, 0],
    amplitude_per_group=[50, 5],
    PLOT=False,
)

model_params = {
    "n_forecasts": 1,
    "epochs": 20,
    "global_normalization": True,
    "normalize": "off",
    "n_changepoints": 0,
    "growth": "off",
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": True,
    "n_lags": 24,
}

run(df=df,
    df_name="0506_SEA_SHAPE_unbalanced_twhiam_2_1_n_lags",
    freq="H",
    model_class=NeuralProphetModel,
    save=True,
    dir_name="0506_SEA_SHAPE_unbalanced_twhiam_2_1_n_lags",
    model_params=model_params)
