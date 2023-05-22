import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly_resampler import unregister_plotly_resampler

unregister_plotly_resampler()
from abc import ABCMeta

import plotly.io as pio
from pandas import Index
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from statsmodels.tsa.arima_process import ArmaProcess

from tot.error_utils import raise_if
from tot.evaluation.metric_utils import calculate_metrics_by_ID_for_forecast_step
from tot.plotting import plot_plotly


class LogTransformer(FunctionTransformer):
    def __init__(self):
        self.epsilon = 1e-8
        super().__init__(self._log_transform, validate=False)

    def _log_transform(self, X):
        return np.log1p(np.maximum(X, self.epsilon))


def load_EIA(n_samples=None, ids=None, n_ids=None):
    datasets_dir = os.path.join(Path(__file__).parent.parent.absolute(), "datasets")
    df = pd.read_csv(datasets_dir + "/eia_electricity_hourly.csv")

    raise_if(
        ids is not None and n_ids is not None, "Remove specified ids from input if you want to select a number of ids."
    )
    if ids is None and n_ids is not None:
        unique_ids = df["ID"].unique()
        ids = random.sample(list(unique_ids), k=10)

    if ids is not None:
        df = df[df["ID"].isin(ids)].reset_index(drop=True)
    if n_samples is not None:
        df = df.groupby("ID").apply(lambda x: x.iloc[:n_samples, :].copy(deep=True)).reset_index(drop=True)
    return df


def load_London(n_samples=None, ids=None, n_ids=None):
    datasets_dir = os.path.join(Path(__file__).parent.parent.absolute(), "datasets")
    df = pd.read_csv(datasets_dir + "/london_electricity_hourly.csv")

    raise_if(
        ids is not None and n_ids is not None, "Remove specified ids from input if you want to select a number of ids."
    )
    if ids is None and n_ids is not None:
        unique_ids = df["ID"].unique()
        ids = random.sample(list(unique_ids), k=10)

    if ids is not None:
        df = df[df["ID"].isin(ids)].reset_index(drop=True)
    if n_samples is not None:
        df = df.groupby("ID").apply(lambda x: x.iloc[:n_samples, :].copy(deep=True)).reset_index(drop=True)
    return df


def load_ERCOT(n_samples=None, ids=None, n_ids=None):
    datasets_dir = os.path.join(Path(__file__).parent.parent.absolute(), "datasets")
    df = pd.read_csv(datasets_dir + "/ercot_load_reduced.csv")

    raise_if(
        ids is not None and n_ids is not None, "Remove specified ids from input if you want to select a number of ids."
    )
    if ids is None and n_ids is not None:
        unique_ids = df["ID"].unique()
        ids = random.sample(list(unique_ids), k=10)

    if ids is not None:
        df = df[df["ID"].isin(ids)].reset_index(drop=True)
    if n_samples is not None:
        df = df.groupby("ID").apply(lambda x: x.iloc[:n_samples, :].copy(deep=True)).reset_index(drop=True)
    return df


def load_Australian(n_samples=None, ids=None, n_ids=None):
    datasets_dir = os.path.join(Path(__file__).parent.parent.absolute(), "datasets")
    df = pd.read_csv(datasets_dir + "/australian_electricity_half_hourly.csv")

    raise_if(
        ids is not None and n_ids is not None, "Remove specified ids from input if you want to select a number of ids."
    )
    if ids is None and n_ids is not None:
        unique_ids = df["ID"].unique()
        ids = random.sample(list(unique_ids), k=10)

    if ids is not None:
        df = df[df["ID"].isin(ids)].reset_index(drop=True)
    if n_samples is not None:
        df = df.groupby("ID").apply(lambda x: x.iloc[:n_samples, :].copy(deep=True)).reset_index(drop=True)
    return df


def load_Solar(n_samples=None, ids=None, n_ids=None):
    datasets_dir = os.path.join(Path(__file__).parent.parent.absolute(), "datasets")
    df = pd.read_csv(datasets_dir + "/solar_10_minutes_dataset.csv")

    raise_if(
        ids is not None and n_ids is not None, "Remove specified ids from input if you want to select a number of ids."
    )
    if ids is None and n_ids is not None:
        unique_ids = df["ID"].unique()
        ids = random.sample(list(unique_ids), k=10)

    if ids is not None:
        df = df[df["ID"].isin(ids)].reset_index(drop=True)
    if n_samples is not None:
        df = df.groupby("ID").apply(lambda x: x.iloc[:n_samples, :].copy(deep=True)).reset_index(drop=True)
    return df


def generate_canceling_shape_season_data(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
) -> pd.DataFrame:
    df_seasons = []
    period = 24
    # Generate an array of time steps
    t = np.arange(series_length)
    # Define the angular frequency (omega) corresponding to the period
    omega = 2 * np.pi / period
    # Generate the seasonal time series using multiple sine and cosine terms
    data_group_1 = [
        (np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t)) * amplitude_per_group[0]
        for _ in range(n_ts_groups[0])
    ]
    for i in range(n_ts_groups[0]):
        df = pd.DataFrame(date_rng, columns=["ds"])
        df["y"] = data_group_1[i] + offset_per_group[0]
        df["ID"] = str(i)
        df_seasons.append(df.reset_index(drop=True))
    i = i
    data_group_2 = [
        -(np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t))
        * amplitude_per_group[1]
        for _ in range(n_ts_groups[1])
    ]
    for j in range(n_ts_groups[1]):
        df = pd.DataFrame(date_rng, columns=["ds"])
        df["y"] = data_group_2[j] + offset_per_group[1]
        df["ID"] = str(i + j + 1)
        df_seasons.append(df.reset_index(drop=True))

    concatenated_dfs = pd.DataFrame()
    for i, df in enumerate(df_seasons):
        concatenated_dfs = pd.concat([concatenated_dfs, df], axis=0)
    return concatenated_dfs


def generate_one_shape_season_data(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
) -> pd.DataFrame:
    df_seasons = []
    period = 24
    # Generate an array of time steps
    t = np.arange(series_length)
    # Define the angular frequency (omega) corresponding to the period
    omega = 2 * np.pi / period
    # Generate the seasonal time series using multiple sine and cosine terms
    data_group_1 = [
        (np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t)) * amplitude_per_group[0]
        for _ in range(n_ts_groups[0])
    ]
    for i in range(n_ts_groups[0]):
        df = pd.DataFrame(date_rng, columns=["ds"])
        df["y"] = data_group_1[i] + offset_per_group[0]
        df["ID"] = str(i)
        df_seasons.append(df.reset_index(drop=True))
    i = i
    data_group_2 = [
        (np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t)) * amplitude_per_group[1]
        for _ in range(n_ts_groups[1])
    ]
    for j in range(n_ts_groups[1]):
        df = pd.DataFrame(date_rng, columns=["ds"])
        df["y"] = data_group_2[j] + offset_per_group[1]
        df["ID"] = str(i + j + 1)
        df_seasons.append(df.reset_index(drop=True))

    concatenated_dfs = pd.DataFrame()
    for i, df in enumerate(df_seasons):
        concatenated_dfs = pd.concat([concatenated_dfs, df], axis=0)

    return concatenated_dfs


def generate_ar(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
) -> pd.DataFrame:
    # Create a DataFrame with the simulated data and date range
    ar_dfs = []
    np.random.seed(42)
    # Define AR coefficients (AR(4) model with coefficients 0.5 and 0.3)
    ar_coeffs = np.array([1, 0.5, -0.3, 0.02, 0.01])
    ma_coeffs = np.array([1])  # MA coefficients (no MA component)
    # Create an ARMA process with the specified coefficients
    ar_process = ArmaProcess(ar_coeffs, ma_coeffs, nobs=series_length)

    ar_data_group_1 = [
        ar_process.generate_sample(series_length) * amplitude_per_group[0] for _ in range(n_ts_groups[0])
    ]
    ar_data_norm_goup1 = [(ar_data_group_1[i] - np.mean(ar_data_group_1[i])) for i in range(n_ts_groups[0])]

    for i in range(n_ts_groups[0]):
        simulated_df = pd.DataFrame(date_rng, columns=["ds"])
        simulated_df["y"] = ar_data_norm_goup1[i] + offset_per_group[0]
        simulated_df["ID"] = str(i)
        ar_dfs.append(simulated_df)

    ar_data_group_2 = [
        ar_process.generate_sample(series_length) * amplitude_per_group[1] for _ in range(n_ts_groups[1])
    ]
    ar_data_norm_goup2 = [(ar_data_group_2[i] - np.mean(ar_data_group_2[i])) for i in range(n_ts_groups[1])]

    for j in range(n_ts_groups[1]):
        simulated_df = pd.DataFrame(date_rng, columns=["ds"])
        simulated_df["y"] = ar_data_norm_goup2[j] + offset_per_group[1]
        simulated_df["ID"] = str(i + 1 + j)
        ar_dfs.append(simulated_df)

    concatenated_dfs = pd.DataFrame()
    for i, df in enumerate(ar_dfs):
        concatenated_dfs = pd.concat([concatenated_dfs, df], axis=0)
    return concatenated_dfs


def layout():
    layout_args = {
        "autosize": True,
        "template": "plotly_white",
        "margin": go.layout.Margin(l=0, r=10, b=0, t=10, pad=0),
        "font": dict(size=10),
        "title": dict(font=dict(size=12)),
        "hovermode": "x unified",
    }
    xaxis_args = {
        "showline": True,
        "mirror": True,
        "linewidth": 1.5,
    }
    yaxis_args = {
        "showline": True,
        "mirror": True,
        "linewidth": 1.5,
    }
    prediction_color = "black"
    actual_color = "black"
    trend_color = "#B23B00"
    line_width = 2
    marker_size = 4
    figsize = (900, 450)
    layout = go.Layout(
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        xaxis=go.layout.XAxis(
            type="date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            **xaxis_args,
        ),
        yaxis=go.layout.YAxis(**yaxis_args),
        **layout_args,
        paper_bgcolor="rgba(250,250,250,250)",
        plot_bgcolor="rgba(250,250,250,250)",
    )
    return layout


def plot_ts(df):
    fig = go.Figure()
    for group_name, group_data in df.groupby("ID"):
        fig.add_trace(
            go.Scatter(
                x=group_data["ds"],
                y=group_data["y"],
                mode="lines",
                name=group_name,
            )
        )
    fig.update_layout(layout())
    return fig


def plot_and_save(df, plot=True, save=False, file_name=None):
    fig = plot_ts(df)
    if plot:
        fig.show()
    if save:
        fig.write_image(file_name)


def gen_model_and_params(common_params, model_class, scalers, scaling_levels, weighted_loss):
    # Add desried scalers here
    if scalers == "default":
        # Note: Box-Cox can only be applied to strictly positive data
        # Note: LogTransformer can only be applied to positive data
        scalers = [
            StandardScaler(),
            MinMaxScaler(feature_range=(-1, 1)),
            MinMaxScaler(feature_range=(0, 1)),
            RobustScaler(quantile_range=(25, 75)),
            # PowerTransformer(method='box-cox', standardize=True),
            # PowerTransformer(method='yeo-johnson', standardize=True),
            QuantileTransformer(output_distribution="normal"),
            # LogTransformer(),
        ]
    if scaling_levels == "default":
        scaling_levels = ["per_time_series", "per_dataset"]
    if weighted_loss is True:
        weighted_loss = ["none", "std*avg", "avg", "std"]
    else:
        weighted_loss = ["none"]
    model_classes_and_params = [(model_class, common_params)]
    for scaler in scalers:
        for scaling_level in scaling_levels:
            if scaling_level == "per_time_series":
                for weighting in weighted_loss:
                    params = common_params.copy()
                    params.update({"scaler": scaler, "scaling_level": scaling_level, "weighted_loss": weighting})
                    model_classes_and_params.append((model_class, params))
            else:
                params = common_params.copy()
                params.update({"scaler": scaler, "scaling_level": scaling_level})
                model_classes_and_params.append((model_class, params))
    model_classes_and_params.append((model_class, params))
    return model_classes_and_params


def save_params(params, dir, df_name, save=True):
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, ABCMeta):
                return obj.__name__
            if isinstance(obj, StandardScaler):
                return str(obj)
            if isinstance(obj, MinMaxScaler):
                return str(obj)
            if isinstance(obj, RobustScaler):
                return str(obj)
            if isinstance(obj, LogTransformer):
                return str(obj)
            if isinstance(obj, PowerTransformer):
                return str(obj)
            if isinstance(obj, QuantileTransformer):
                return str(obj)
            if isinstance(obj, FunctionTransformer):
                return str(obj)
            return super().default(obj)

    if save:
        config_file_name = os.path.join(dir, f"{df_name}.json")
        with open(config_file_name, "w") as file:
            json.dump(params, file, cls=CustomJSONEncoder)


def plot_forecast(df, plot=True, save=True, file_name=None, file_name_fcst=None):
    fig = go.Figure()

    unique_ids = df["ID"].unique()
    n_ids = len(unique_ids)
    if n_ids > 10:
        sampled_ids = random.sample(list(unique_ids), k=10)
        n_ids = len(sampled_ids)
    else:
        sampled_ids = unique_ids
    # fig_all = plotly.subplots.make_subplots(rows=n_ids, cols=1)
    figs = []
    for i, fcst_df in df[df["ID"].isin(sampled_ids)].groupby("ID"):
        fig = plot_plotly(
            fcst=fcst_df,
            df_name="none",
            xlabel="ds",
            ylabel="y",
            # highlight_forecast=None,
            figsize=(700, 350),
            plotting_backend="plotly",
        )
        figs.append(fig)

    if plot:
        for fig in figs:
            fig.show()

    if save:
        assert n_ids == len(figs), "ids and figs must have the same length"
        figure = plotly.subplots.make_subplots(rows=n_ids, cols=1)
        for i, fig in enumerate(figs):
            figure.add_trace(fig.data[0], row=i + 1, col=1)
            figure.add_trace(fig.data[1], row=i + 1, col=1)

        x_range = figure.data[0].x
        # Get the last 100 dates from the x-axis data
        if len(x_range) > 200:
            x_range_limit = x_range[-200:]
            figure.update_xaxes(range=[x_range_limit[0], x_range_limit[-1]])
        figure.update_layout(autosize=False, width=500, height=1400)
        pio.write_image(figure, file_name)
        del figure
        fcst_df.to_csv(file_name_fcst)


def plot_forecasts(benchmark, dir, plot=False, save=True):
    for exp, fcst_test in zip(benchmark.experiments, benchmark.fcst_test):
        plot_file_name = os.path.join(dir, f"{exp.experiment_name}.png")
        file_name_fcst = os.path.join(dir, f"{exp.experiment_name}.csv")
        plot_forecast(fcst_test, plot=plot, save=save, file_name=plot_file_name, file_name_fcst=file_name_fcst)


def save_results(benchmark, metrics, freq, dir, save):
    if save:
        # result details
        for i, (fcst_test, fcst_train, exp) in enumerate(
            zip(benchmark.fcst_test, benchmark.fcst_train, benchmark.experiments)
        ):
            df_metric_perID = calculate_metrics_by_ID_for_forecast_step(
                fcst_df=fcst_test,
                df_historic=fcst_train,
                metrics=metrics,
                forecast_step_in_focus=None,
                freq=freq,
            )
            df_metric_perID.index = df_metric_perID.index.astype(str)
            df_metrics_sum = pd.DataFrame(benchmark.df_metrics_test.loc[[i], metrics])
            df_metrics_sum.index = Index(["ALL"], name="ID")
            df_metric_perID = pd.concat([df_metric_perID, df_metrics_sum], axis=0)

            file_name = os.path.join(dir, f"results_{exp.experiment_name}.csv")
            df_metric_perID.to_csv(file_name)

        # result summary
        metric_dfs = benchmark.df_metrics_test
        elapsed_times = benchmark.elapsed_times
        results_file_name = os.path.join(dir, "results.csv")
        metric_dfs.to_csv(results_file_name)

        df_elapsed_time = pd.DataFrame(elapsed_times, columns=["elapsed_times"])
        elapsed_time_file_name = os.path.join(dir, "elapsed_time.csv")
        df_elapsed_time.to_csv(elapsed_time_file_name)
