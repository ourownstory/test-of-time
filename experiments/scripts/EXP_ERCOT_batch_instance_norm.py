from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from experiments.pipeline_experiment import run
from experiments.utils import LogTransformer, load_ERCOT, gen_model_and_params_norm
from tot.models import NeuralProphetModel

PLOT = False
DIR_NAME = "EXP_ERCOT_batch_instance_norm"
FREQ = "H"
MODEL = NeuralProphetModel
MODEL_PARAMS = {
    "n_forecasts": 1,
    "epochs": 100,
    "global_normalization": True,
    "trend_global_local": "global",
    "season_global_local": "global",
    "normalize": "off",
    "n_lags": 24,
}

run(
    dir_name=DIR_NAME,
    save=True,
    df=load_ERCOT(),
    df_name=DIR_NAME,
    freq=FREQ,
    model_class=MODEL,
    model_params=MODEL_PARAMS,
    scalers=["batch", "instance"],
    metrics=["MAE", "RMSE", "MAPE", "MASE"],
    test_percentage=0.25,
    plot=PLOT,
    model_and_params_generator=gen_model_and_params_norm
)
