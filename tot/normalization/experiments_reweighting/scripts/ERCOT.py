from tot.models import NeuralProphetModel
from tot.normalization.experiments_reweighting.pipeline_reweighting import run
from tot.normalization.experiments_reweighting.utils import get_ERCOT, filter_ID

df_global = get_ERCOT()

ERCOT_full = (df_global, "full")
similar_season_diff_amplitude_2_1 = (
    filter_ID(df_global, ["WEST", "FAR_WEST", "NORTH_C"]), "similar_season_diff_amplitude_2_1")
similar_season_diff_amplitude_1_1 = (filter_ID(df_global, ["NORTH_C", "WEST"]), "similar_season_diff_amplitude_1_1")

data = [ERCOT_full, similar_season_diff_amplitude_2_1, similar_season_diff_amplitude_1_1]

model_params = {
    "n_forecasts": 1,
    "epochs": 20,
    "global_normalization": True,
    "normalize": "off",
    "n_lags": 24,
}

for df, name in data:
    run(df=df,
        df_name=name,
        freq="H",
        model_class=NeuralProphetModel,
        save=True,
        dir_name="ERCOT",
        model_params=model_params)
