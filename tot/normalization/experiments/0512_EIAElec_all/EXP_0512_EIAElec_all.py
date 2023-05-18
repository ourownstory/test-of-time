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
EXP_NAME = "0512_EIAElec_all"
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, "plots_NeuralProphetModel")
PLOT = False

SERIES_LENGTH = 24 * 7 * 15
# DATE_RNG = date_rng = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
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
EIA_ELEC_FILE = os.path.join(_DEFAULT_DIRECTORY, "eia_electricity_hourly.csv")
df_eia_elec = pd.read_csv(EIA_ELEC_FILE)
df_eia_elec = df_eia_elec.drop('Unnamed: 0', axis=1)
df_eia_elec = df_eia_elec.sort_values(['ID', 'ds']).groupby('ID').apply(lambda x: x[0:SERIES_LENGTH]).reset_index(drop=True)
min_date = df_eia_elec[df_eia_elec['ID']=='DOPD'].loc[:, 'ds'].min()
max_date = df_eia_elec[df_eia_elec['ID']=='DOPD'].loc[:, 'ds'].max()
DATE_RNG = pd.date_range(start=min_date, end=max_date, freq="H")
fig = plot_ts(df_eia_elec)
if PLOT:
    fig.show()
file_name = os.path.join(PLOTS_DIR, f"DATA_{EXP_NAME}.png")
pio.write_image(fig, file_name)

start_time = time.time()

fcsts_train, fcsts_test, metrics_test, elapsed_time = run_pipeline(
    df=df_eia_elec,
    model_class=MODEL_CLASS,
    params=PARAMS,
    freq="30min",
    test_percentage=0.4,
    metrics=["MAPE", "MAE", "RMSE", "MASE"],
    scale_levels=[None, "local", "global"],
    scalers=[MinMaxScaler(feature_range=(0, 1)), StandardScaler(), MaxAbsScaler()],
)
plot_and_save_multiple_dfs_multiple_ids(
    fcst_dfs=fcsts_test,
    date_rng=DATE_RNG,
    ids=['AEC', 'AECI', 'AVA','AZPS', 'BANC', 'BPAT', 'CHPD', 'CISO', 'CPLE', 'CPLW', 'DOPD'],
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