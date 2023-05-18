import os
import pathlib
import time
from pathlib import Path
import plotly.io as pio
import pandas as pd
from neuralprophet import set_log_level
from plotly_resampler import unregister_plotly_resampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

from tot.models.models_neuralprophet import NeuralProphetModel
from tot.normalization.experiments.pipeline import (
    concat_and_save_results,
    plot_and_save_multiple_dfs_multiple_ids,
    run_pipeline,
    plot_ts,
)

unregister_plotly_resampler()


set_log_level("INFO")
DIR = pathlib.Path(__file__).parent.parent.absolute()
EXP_NAME = "0512_LondonElec_all"
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, f"plots_NeuralProphetModel")
PLOT = False

SERIES_LENGTH = 24 * 7 * 15
MODEL_CLASS = NeuralProphetModel
PARAMS = {
    "n_forecasts": 1,
    "n_lags": 24,
    # "n_changepoints": 0,
    # "growth": "off",
    "global_normalization": True,
    "normalize": "off",
    # Disable seasonality components, except yearly
    # "yearly_seasonality": False,
    # "weekly_seasonality": False,
    # "daily_seasonality": True,
    "epochs": 20,
    "_data_params": {},
}
_DEFAULT_DIRECTORY = os.path.join(Path(__file__).parent.parent.parent.parent.parent.absolute(), "datasets")
LONDON_FILE = os.path.join(_DEFAULT_DIRECTORY, "london_electricity_hourly.csv")
df_london = pd.read_csv(LONDON_FILE)
df_london = df_london.drop('Unnamed: 0', axis=1)
df_london = df_london.sort_values(['ID', 'ds']).groupby('ID').apply(lambda x: x[0:SERIES_LENGTH]).reset_index(drop=True)
df_london = df_london[df_london['ID'].isin(['T1', 'T26', 'T59', 'T98', 'T134', 'T179', 'T202', 'T241','T292', 'T320'])].reset_index(drop=True)
min_date = df_london[df_london['ID']=='T1'].loc[:, 'ds'].min()
max_date = df_london[df_london['ID']=='T1'].loc[:, 'ds'].max()
DATE_RNG = pd.date_range(start=min_date, end=max_date, freq="H")
fig = plot_ts(df_london)
if PLOT:
    fig.show()
file_name = os.path.join(PLOTS_DIR, f"DATA_{EXP_NAME}.png")
pio.write_image(fig, file_name)
start_time = time.time()

fcsts_train, fcsts_test, metrics_test, elapsed_time = run_pipeline(
    df=df_london,
    model_class=MODEL_CLASS,
    params=PARAMS,
    freq="H",
    test_percentage=0.4,
    metrics=["MAPE", "MAE", "RMSE", "MASE"],
    scale_levels=[None, "local", "global"],
    scalers=[MinMaxScaler(feature_range=(0, 1)), StandardScaler(), MaxAbsScaler()],
)
plot_and_save_multiple_dfs_multiple_ids(
    fcst_dfs=fcsts_test,
    date_rng=DATE_RNG,
    ids=['T1', 'T26', 'T59', 'T98', 'T134', 'T179', 'T202', 'T241','T292', 'T320'],
    PLOT=PLOT,
    PLOTS_DIR=PLOTS_DIR,
    EXP_NAME=EXP_NAME,
)
concat_and_save_results(
    metric_dfs=metrics_test,
    elapsed_time=elapsed_time,
    EXP_DIR=EXP_DIR,
    EXP_NAME=EXP_NAME,
)

end_time = time.time()
elapsed_time = end_time - start_time
print('elapsed_time: ', elapsed_time)