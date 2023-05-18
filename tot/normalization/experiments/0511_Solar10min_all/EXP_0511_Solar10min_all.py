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
EXP_NAME = "0511_Solar10min_all"
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, f"plots_NeuralProphetModel")
PLOT = False

SERIES_LENGTH = 6* 24 * 7 * 3
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
_DEFAULT_DIRECTORY = os.path.join(Path(__file__).parent.parent.parent.parent.parent.absolute(), "ar_data")
SOLAR_FILE = os.path.join(_DEFAULT_DIRECTORY, "solar_10_minutes_dataset.csv")
df_solar = pd.read_csv(SOLAR_FILE)
df_solar = df_solar.drop('Unnamed: 0', axis=1)
df_solar = df_solar.sort_values(['ID', 'ds']).groupby('ID').apply(lambda x: x[0:SERIES_LENGTH]).reset_index(drop=True)
df_solar[df_solar['ID'].isin(['T1', 'T14', 'T26', 'T59', 'T71', 'T98', 'T102', 'T110', 'T122','T134'])].reset_index(drop=True)
min_date = df_solar[df_solar['ID']=='T1'].loc[:, 'ds'].min()
max_date = df_solar[df_solar['ID']=='T1'].loc[:, 'ds'].max()
DATE_RNG = pd.date_range(start=min_date, end=max_date, freq="10min")
fig = plot_ts(df_solar)
if PLOT:
    fig.show()
file_name = os.path.join(PLOTS_DIR, f"DATA_{EXP_NAME}.png")
pio.write_image(fig, file_name)
start_time = time.time()

fcsts_train, fcsts_test, metrics_test, elapsed_time = run_pipeline(
    df=df_solar,
    model_class=MODEL_CLASS,
    params=PARAMS,
    freq="10min",
    test_percentage=0.4,
    metrics=["MAPE", "MAE", "RMSE", "MASE"],
    scale_levels=[None, "local", "global"],
    scalers=[MinMaxScaler(feature_range=(0, 1)), StandardScaler(), MaxAbsScaler()],
)
plot_and_save_multiple_dfs_multiple_ids(
    fcst_dfs=fcsts_test,
    date_rng=DATE_RNG,
    ids=['T1', 'T14', 'T26', 'T59', 'T71', 'T98', 'T102', 'T110', 'T122','T134'],
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