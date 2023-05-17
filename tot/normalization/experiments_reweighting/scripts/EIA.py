from tot.models import NeuralProphetModel
from tot.normalization.experiments_reweighting.pipeline_reweighting import run
from tot.normalization.experiments_reweighting.utils import get_EIA

eia_df = get_EIA()

model_params = {
    "n_forecasts": 1,
    "epochs": 20,
    "global_normalization": True,
    "normalize": "off",
    "n_lags": 24,
}

run(df=eia_df,
    df_name="EIA",
    freq="H",
    model_class=NeuralProphetModel,
    save=True,
    dir_name="EIA",
    model_params=model_params)
