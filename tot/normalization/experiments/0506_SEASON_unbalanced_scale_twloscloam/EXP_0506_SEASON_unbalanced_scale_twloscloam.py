import pandas as pd
import numpy as np
import plotly.subplots
from pandas import Index
import time
import os
import pathlib
import plotly.io as pio
from neuralprophet import set_log_level
from tot.evaluation.metric_utils import calculate_metrics_by_ID_for_forecast_step
from tot.models.models_neuralprophet import NeuralProphetModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tot.plotting import plot_plotly
from tot.normalization.experiments.pipeline import run_custom_pipline, plot_ts
from plotly_resampler import unregister_plotly_resampler
unregister_plotly_resampler()

def generate_one_shape_season_data(series_length:int, date_rng, n_ts_groups:list, offset_per_group: list, amplitude_per_group: list)-> pd.DataFrame:
    df_seasons = []
    period = 24
    # Generate an array of time steps
    t = np.arange(series_length)
    # Define the angular frequency (omega) corresponding to the period
    omega = 2 * np.pi / period
    # Generate the seasonal time series using multiple sine and cosine terms
    data_group_1 = [(np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t)) * amplitude_per_group[0] for _ in range(n_ts_groups[0])]
    for i in range(n_ts_groups[0]):
        df = pd.DataFrame(date_rng, columns=['ds'])
        df['y'] = data_group_1[i] + offset_per_group[0]
        df['ID'] = str(i)
        df_seasons.append(df.reset_index(drop=True))
    i=i
    data_group_2 = [(np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) +
                  np.cos(2 * omega * t)) * amplitude_per_group[1] for _ in range(n_ts_groups[1])]
    for j in range(n_ts_groups[1]):
        df = pd.DataFrame(date_rng, columns=['ds'])
        df['y'] = data_group_2[j] + offset_per_group[1]
        df['ID'] = str(i+j+1)
        df_seasons.append(df.reset_index(drop=True))

    concatenated_dfs = pd.DataFrame()
    for i, df in enumerate(df_seasons):
        concatenated_dfs = pd.concat([concatenated_dfs, df], axis=0)
    fig = plot_ts(concatenated_dfs)
    if PLOT:
        fig.show()
    fig.update_xaxes(range=[date_rng[0], date_rng[24*7]])

    concatenated_dfs.to_csv(os.path.join(PLOTS_DIR, f'DATA_{EXP_NAME}.csv'))
    file_name = os.path.join(PLOTS_DIR, f"DATA_{EXP_NAME}.png")
    pio.write_image(fig, file_name)

    return concatenated_dfs

def run_pipeline(df, model_class, params, freq, test_percentage, metrics, scale_levels:list, scalers:list):
    elapsed_time = pd.DataFrame(columns=['scaler', 'scale_level', 'time'])
    fcsts_train = []
    fcsts_test = []
    metrics_test = []
    for scale_level in scale_levels:
        if scale_level is None:
            scaler = None
            start_time = time.time()
            results_train, results_test, fcst_train, fcst_test = run_custom_pipline(
                df=df, model_class=model_class,
                params=params, freq=freq, test_percentage=test_percentage,
                metrics=metrics, scale_level=scale_level, scaler=scaler)
            end_time = time.time()
            elapsed_time_single = pd.DataFrame(columns=['scaler', 'scale_level', 'time'])
            elapsed_time_single.loc[0] = [str(scaler), str(scale_level), (end_time - start_time)]
            elapsed_time = pd.concat([elapsed_time, elapsed_time_single], axis=0)

            df_metric_perID = calculate_metrics_by_ID_for_forecast_step(
                fcst_df=fcst_test,
                df_historic=fcst_train,
                metrics=metrics,
                forecast_step_in_focus=None,
                freq=freq,
            )
            df_metric_perID.index = df_metric_perID.index.astype(str)
            df_metric_perID = pd.concat([df_metric_perID, pd.DataFrame(
                results_test, index=Index(['ALL'], name='ID'))], axis=0)
            df_metric_perID['scaler'] = scaler
            df_metric_perID['scale_level'] = scale_level
            metrics_test.append(df_metric_perID)
            fcst_train['scaler'] = scaler
            fcst_train['scale_level'] = scale_level
            fcst_test['scaler'] = scaler
            fcst_test['scale_level'] = scale_level
            fcsts_train.append(fcst_train)
            fcsts_test.append(fcst_test)
        else:
            for scaler in scalers:
                start_time = time.time()
                results_train, results_test, fcst_train, fcst_test = run_custom_pipline(
                    df=df, model_class=model_class,
                    params=params, freq=freq, test_percentage=test_percentage,
                    metrics=metrics, scale_level=scale_level, scaler=scaler)
                end_time = time.time()
                elapsed_time_single = pd.DataFrame(columns=['scaler', 'scale_level', 'time'])
                elapsed_time_single.loc[0]=[str(scaler), str(scale_level), (end_time-start_time)]
                elapsed_time = pd.concat([elapsed_time, elapsed_time_single], axis=0)

                df_metric_perID = calculate_metrics_by_ID_for_forecast_step(
                    fcst_df=fcst_test,
                    df_historic=fcst_train,
                    metrics=metrics,
                    forecast_step_in_focus=None,
                    freq=freq,
                )
                df_metric_perID.index = df_metric_perID.index.astype(str)
                df_metric_perID = pd.concat([df_metric_perID, pd.DataFrame(
                    results_test, index=Index(['ALL'], name='ID'))], axis=0)
                df_metric_perID['scaler'] = scaler
                df_metric_perID['scale_level'] = scale_level
                metrics_test.append(df_metric_perID)
                fcst_train['scaler'] = scaler
                fcst_train['scale_level'] = scale_level
                fcst_test['scaler'] = scaler
                fcst_test['scale_level'] = scale_level
                fcsts_train.append(fcst_train)
                fcsts_test.append(fcst_test)

    return fcsts_train, fcsts_test, metrics_test, elapsed_time

def plot_and_save_multiple_dfs(fcst_dfs: list, id_group_1, id_group_2, date_rng):
    for fcst_df in fcst_dfs:
        fig1 = plot_plotly(
            fcst=fcst_df[fcst_df['ID'] == str(id_group_1)],
            df_name='none',
            xlabel="ds",
            ylabel="y",
            # highlight_forecast=None,
            figsize=(700, 350),
            plotting_backend='plotly',
        )
        fig2 = plot_plotly(
            fcst=fcst_df[fcst_df['ID'] == str(id_group_2)],
            df_name='none',
            xlabel="ds",
            ylabel="y",
            # highlight_forecast=None,
            figsize=(700, 350),
            plotting_backend='plotly',
        )
        if PLOT:
            fig1.show()
            fig2.show()
        # join figs
        fig = plotly.subplots.make_subplots(rows=2, cols=1)
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig1.data[1], row=1, col=1)
        fig.add_trace(fig2.data[0], row=2, col=1)
        fig.add_trace(fig2.data[1], row=2, col=1)
        fig.update_xaxes(range=[date_rng[24*7*14], date_rng[(24*7*15)-1]])
        file_name = os.path.join(PLOTS_DIR, f"{fcst_df['scaler'][0]}_{fcst_df['scale_level'][0]}.png")
        pio.write_image(fig, file_name)

def concat_and_save_results(metric_dfs: list, elapsed_time: pd.DataFrame):
    df_metrics_concatenated = pd.DataFrame()
    for metric_df in metric_dfs:
        df_metrics_concatenated = pd.concat([df_metrics_concatenated, metric_df])
    df_metrics_concatenated.to_csv(os.path.join(EXP_DIR, f'RESULTS{EXP_NAME}.csv'))
    elapsed_time.to_csv(os.path.join(EXP_DIR,f"ELAPSED_TIME_{EXP_NAME}.csv"))


set_log_level("INFO")
DIR = pathlib.Path(__file__).parent.parent.absolute()
EXP_NAME = '0506_SEASON_unbalanced_scale_twloscloam'
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
df = generate_one_shape_season_data(series_length=SERIES_LENGTH, date_rng=DATE_RNG, n_ts_groups=[10, 1], offset_per_group=[10, 100], amplitude_per_group=[5, 50])
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

