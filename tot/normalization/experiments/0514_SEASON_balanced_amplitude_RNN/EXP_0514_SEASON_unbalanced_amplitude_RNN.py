import os
import pathlib
import time
import pandas as pd
from neuralprophet import set_log_level
from plotly_resampler import unregister_plotly_resampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from darts.models.forecasting.rnn_model import RNNModel
from tot.models.models_darts import DartsForecastingModel
from tot.normalization.experiments.pipeline import (
    concat_and_save_results,
    generate_one_shape_season_data,
    plot_and_save_multiple_dfs,
    run_pipeline,
)

unregister_plotly_resampler()


set_log_level("INFO")
DIR = pathlib.Path(__file__).parent.parent.absolute()
EXP_NAME = "0514_SEASON_balanced_amplitude_RNN"
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, f"plots")
PLOT = False

SERIES_LENGTH = 24 * 7 * 15
DATE_RNG = date_rng = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
MODEL_CLASS = DartsForecastingModel
PARAMS = {
    "model": RNNModel,
    "input_chunk_length": 24,
    'hidden_dim':20,
    'batch_size':16,
    'n_epochs':20,
    'random_state':0,
    'training_length':24,
    'force_reset':True,
    'lags': 24,
    'n_forecasts': 1,
    '_data_params':{},
}
start_time= time.time()
df = generate_one_shape_season_data(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[5, 5],
    offset_per_group=[0, 0],
    amplitude_per_group=[50, 5],
    PLOT=PLOT,
    PLOTS_DIR=PLOTS_DIR,
    EXP_NAME=EXP_NAME,
)
fcsts_train, fcsts_test, metrics_test, elapsed_time = run_pipeline(
    df=df,
    model_class=MODEL_CLASS,
    params=PARAMS,
    freq="H",
    test_percentage=0.4,
    metrics=["MAPE", "MAE", "RMSE", "MASE"],
    scale_levels=[None, "local", "global"],
    scalers=[StandardScaler(),MinMaxScaler(feature_range=(-1, 1))],
)
plot_and_save_multiple_dfs(
    fcst_dfs=fcsts_test,
    date_rng=DATE_RNG,
    id_group_1=str(1),
    id_group_2=str(5),
    PLOT=PLOT,
    PLOTS_DIR=PLOTS_DIR,
    EXP_NAME=EXP_NAME,
)
concat_and_save_results(metric_dfs=metrics_test, elapsed_time=elapsed_time, EXP_DIR=EXP_DIR, EXP_NAME=EXP_NAME)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")