import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from experiments.pipeline_experiment import run
from experiments.utils import generate_intermittent_multiple_shapes, LogTransformer, ShiftedBoxCoxTransformer
from tot.models import NeuralProphetModel

PLOT = False
DIR_NAME = "INTERMITTENT_SEA_SHAPE_unbalanced_scale_NP.py"
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

df = generate_intermittent_multiple_shapes(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[1, 2],
    amplitude_per_group=[50, 5],
)

scalers = [
    StandardScaler(),
    MinMaxScaler(feature_range=(-1, 1)),
    MinMaxScaler(feature_range=(0, 1)),
    RobustScaler(quantile_range=(25, 75)),
    # PowerTransformer(method="box-cox", standardize=True),
    ShiftedBoxCoxTransformer(),
    PowerTransformer(method="yeo-johnson", standardize=True),
    QuantileTransformer(output_distribution="normal"),
    LogTransformer(),
]

run(
    dir_name=DIR_NAME,
    save=True,
    df=df,
    df_name=DIR_NAME,
    freq=FREQ,
    model_class=MODEL,
    model_params=MODEL_PARAMS,
    scalers=scalers,
    scaling_levels="default",
    reweight_loss=True,
    metrics=["MAE", "RMSE", "MAPE", "MASE"],
    test_percentage=0.25,
    plot=PLOT,
)
