import os
import pathlib
import pandas as pd
from neuralprophet import set_log_level
from plotly_resampler import unregister_plotly_resampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from darts.models.forecasting.transformer_model import TransformerModel
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
EXP_NAME = "0513_SEASON_unbalanced_amplitude_twloam_Transformer"
EXP_DIR = os.path.join(DIR, f"{EXP_NAME}")
PLOTS_DIR = os.path.join(EXP_DIR, f"plots")
PLOT = False

SERIES_LENGTH = 24 * 7 * 15
DATE_RNG = date_rng = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
MODEL_CLASS = DartsForecastingModel
PARAMS = {
    "model": TransformerModel,
    'model_name':'air_transformer',
    "n_forecasts": 1,
    'lags':24,
    "output_chunk_length": 1,
    "input_chunk_length":24,
    'batch_size':32,
    'n_epochs':100,
    'nr_epochs_val_period':10,
    'd_model':16,
    # 'n_heads':8,
    'num_encoder_layers':2,
    'num_decoder_layers':2,
    'dim_feedforward':128,
    'dropout':0.1,
    'activation':'relu',
    'random_state':42,
    'save_checkpoints':True,
    'force_reset':True,
    "_data_params": {},
}
df = generate_one_shape_season_data(
    series_length=SERIES_LENGTH,
    date_rng=DATE_RNG,
    n_ts_groups=[10, 1],
    offset_per_group=[0, 0],
    amplitude_per_group=[5, 50],
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
    id_group_2=str(10),
    PLOT=PLOT,
    PLOTS_DIR=PLOTS_DIR,
    EXP_NAME=EXP_NAME,
)
concat_and_save_results(metric_dfs=metrics_test, elapsed_time=elapsed_time, EXP_DIR=EXP_DIR, EXP_NAME=EXP_NAME)
