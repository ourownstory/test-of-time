import pandas as pd
import os
import pathlib
from neuralprophet import set_log_level
from tot.models.models_neuralprophet import NeuralProphetModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tot.normalization.experiments.pipeline import concat_and_save_results, run_pipeline, generate_one_shape_season_data, plot_and_save_multiple_dfs
from plotly_resampler import unregister_plotly_resampler
unregister_plotly_resampler()



set_log_level("INFO")
DIR = pathlib.Path(__file__).parent.parent.absolute()
EXP_NAME = '0506_SEA_SHAPE_unbalanced'
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, f"plots")
PLOT=True

SERIES_LENGTH = 24*7*15
DATE_RNG=date_rng = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq='H')
MODEL_CLASS = NeuralProphetModel
PARAMS = {
    "n_forecasts": 1,
    "n_changepoints": 0,
    "growth": "off",
    "global_normalization": True,
    "normalize": "minmax",
    # Disable seasonality components, except yearly
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": True,
    "epochs": 20,
    "_data_params": {},
}
df = generate_canceling_shape_season_data(series_length=SERIES_LENGTH, date_rng=DATE_RNG, n_ts_groups=[10, 1], offset_per_group=[0, 0], amplitude_per_group=[5, 5])
fcsts_train, fcsts_test, metrics_test, elapsed_time = run_pipeline(
    df=df,
    model_class=MODEL_CLASS,
    params=PARAMS,
    freq='H',
    test_percentage=0.4,
    metrics=['MAPE', 'MAE', 'RMSE', 'MASE'],
    scale_levels=[None, 'local', 'global'],
    scalers=[MinMaxScaler(feature_range=(0.1, 1)), StandardScaler()]
)
plot_and_save_multiple_dfs(fcst_dfs=fcsts_test, date_rng=DATE_RNG, id_group_1=str(1), id_group_2=str(10))
concat_and_save_results(metric_dfs=metrics_test, elapsed_time=elapsed_time)

