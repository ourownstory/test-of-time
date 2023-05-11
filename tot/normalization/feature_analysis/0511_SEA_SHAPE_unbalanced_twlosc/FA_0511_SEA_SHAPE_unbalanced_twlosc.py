import os
import pathlib

import pandas as pd
import numpy as np
from neuralprophet import set_log_level
from plotly_resampler import unregister_plotly_resampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

from tot.models.models_neuralprophet import NeuralProphetModel
from tot.normalization.experiments.pipeline import (
    generate_canceling_shape_season_data,
    plot_and_save_scaled_dfs,
    data_specific_preprocessing,
    run_scale_pipeline,
)

unregister_plotly_resampler()


set_log_level("INFO")
DIR = pathlib.Path(__file__).parent.parent.absolute()
EXP_NAME = "0511_SEA_SHAPE_unbalanced_twlosc"
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, f"plots")
FREQ='H'
TEST_PERCENTAGE=0.4
PLOT = False

SERIES_LENGTH = 24 * 7 * 15
DATE_RNG = date_rng = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq=FREQ)
MODEL_CLASS = NeuralProphetModel
PARAMS = {
    "n_forecasts": 1,
    "n_changepoints": 0,
    "growth": "off",
    "global_normalization": True,
    "normalize": "off",
    # Disable seasonality components, except yearly
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": True,
    "epochs": 20,
    "_data_params": {},
}
df = generate_canceling_shape_season_data(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[1, 2],
    offset_per_group=[100, 10],
    amplitude_per_group=[5, 5],
    PLOT=PLOT,
    PLOTS_DIR=PLOTS_DIR,
    EXP_NAME=EXP_NAME,

)

df_train, df_test, received_ID_col, received_single_time_series = data_specific_preprocessing(
        df=df, freq=FREQ, test_percentage=TEST_PERCENTAGE
)

dfs_train, dfs_test, fitted_scalers = run_scale_pipeline(
    df_train,
    df_test,
    scalers=[MinMaxScaler(feature_range=(0, 0.5)), StandardScaler(), MaxAbsScaler()],
    scale_level='local'
)

# plot_and_save_scaled_dfs(df_train=df_train, df_test=df_test, dfs_train=dfs_train, dfs_test=dfs_test, date_rng=DATE_RNG, PLOT=PLOT, PLOTS_DIR=PLOTS_DIR)

from scipy.stats import ttest_ind

stat, p_value = ttest_ind(dfs_train['StandardScaler()'].loc[dfs_train['StandardScaler()']['ID']=='0','y'], dfs_train['StandardScaler()'].loc[dfs_train['StandardScaler()']['ID']=='2','y'])
print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")

stat, p_value = ttest_ind(df_train.loc[df_train['ID']=='0','y'], df_train.loc[df_train['ID']=='2','y'])
print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")



# Init dataframe
df_bins = pd.DataFrame()

# Generate bins from control group
values_0 = df_train.loc[df_train['ID']=='0','y']
values_2 = df_train.loc[df_train['ID']=='2','y']
_, bins = pd.qcut(values_0, q=10, retbins=True)
df_bins['bin'] = pd.cut(values_0, bins=bins).value_counts().index

# Apply bins to both groups
df_bins['0_observed'] = pd.cut(df_train.loc[df_train['ID']=='0','y'], bins=bins).value_counts().values
df_bins['2_observed'] = pd.cut(df_train.loc[df_train['ID']=='2','y'], bins=bins).value_counts().values

# Compute expected frequency in the treatment group
df_bins['2_expected'] = df_bins['0_observed'] / np.sum(df_bins['0_observed']) * np.sum(df_bins['2_observed'])

from scipy.stats import chisquare

stat, p_value = chisquare(df_bins['2_observed'], df_bins['2_expected'])
print(f"Chi-squared Test: statistic={stat:.4f}, p-value={p_value:.4f}")
