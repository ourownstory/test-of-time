from tot.models import NeuralProphetModel
from tot.normalization.experiments_reweighting.pipeline_reweighting import run
from tot.normalization.experiments_reweighting.utils import get_ERCOT, filter_ID

df_global = get_ERCOT()

chosen_regions = ["COAST", "NORTH_C", "EAST", "WEST"]
df = filter_ID(df_global, chosen_regions)
df.loc[df["ID"] == "EAST", 'y'] = -1 * df.loc[df["ID"] == "EAST", 'y'] + 2 * df[df["ID"] == "EAST"]['y'].mean()
df.loc[df["ID"] == "WEST", 'y'] = -1 * df.loc[df["ID"] == "WEST", 'y'] + 2 * df[df["ID"] == "WEST"]['y'].mean()

model_params = {
    "n_forecasts": 1,
    "epochs": 20,
    "global_normalization": True,
    "normalize": "off",
    "n_lags": 24,
}

run(df=df,
    df_name="diff_amplitude_canceling_shape",
    freq="H",
    model_class=NeuralProphetModel,
    save=True,
    dir_name="diff_amplitude_canceling_season_2_2-manipulated_ERCOT",
    model_params=model_params)
