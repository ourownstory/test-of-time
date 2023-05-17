import os
from pathlib import Path

import plotly
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def plot(df, show=True, save=False, file_name=None):
    fig = go.Figure()

    for region in df["ID"].unique():
        fig.add_trace(
            go.Scatter(
                name=region,
                x=df[df['ID'] == region]['ds'],
                y=df[df['ID'] == region]['y'])
        )

    if show:
        fig.show()
    if save:
        fig.update_layout(
            width=1800,
            height=1000,
        )
        fig.write_image(file_name)


def filter_ID(df, values):
    return df[df.ID.isin(values)]


def plot_forecast(df, show=True, save=False, file_name=None):
    fig = go.Figure()

    n_rows = len(df["ID"].unique())
    fig_all = plotly.subplots.make_subplots(rows=n_rows, cols=1)

    for i, region in enumerate(df["ID"].unique()):
        trace_yhat = go.Scatter(
            name=region + ' predicted',
            x=df[df['ID'] == region]['ds'],
            y=df[df['ID'] == region]['yhat1'])
        trace_y = go.Scatter(
            name=region + ' actual',
            mode="markers",
            x=df[df['ID'] == region]['ds'],
            y=df[df['ID'] == region]['y'])
        fig.add_trace(trace_yhat)
        fig.add_trace(trace_y)
        fig_all.add_trace(trace_yhat, row=i + 1, col=1)
        fig_all.add_trace(trace_y, row=i + 1, col=1)

    if show:
        fig.show()
        fig_all.show()
    if save:
        fig_all.update_layout(
            width=1200,
            height=300 * n_rows,
        )
        fig_all.write_image(file_name)


def gen_model_and_params(common_params, model_class):
    model_classes_and_params = [(model_class, common_params)]
    for scaler in [StandardScaler(), MinMaxScaler(feature_range=(-1, 1))]:
        for scaling_level in ["per_time_series", "per_dataset"]:
            if scaling_level == "per_time_series":
                for weighting in ["none", "std*avg", "avg", "std"]:
                    params = common_params.copy()
                    params.update(
                        {"scaler": scaler, "scaling_level": scaling_level, "weighted_loss": weighting})
                    model_classes_and_params.append((model_class, params))
            else:
                params = common_params.copy()
                params.update(
                    {"scaler": scaler, "scaling_level": scaling_level})
                model_classes_and_params.append((model_class, params))
    return model_classes_and_params


def get_ERCOT():
    data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"
    df_ercot = pd.read_csv(data_location + "multivariate/load_ercot_regions.csv")
    regions = list(df_ercot)[1:]
    df_global = pd.DataFrame()
    for col in regions:
        aux = df_ercot[["ds", col]].copy(deep=True)
        aux = aux.copy(deep=True)
        aux = aux.iloc[-26000:, :].copy(deep=True)
        aux = aux.rename(columns={col: "y"})
        aux["ID"] = col
        df_global = pd.concat((df_global, aux))
    return df_global


def get_EIA():
    datasets_dir = os.path.join(Path(__file__).parent.parent.parent.parent.absolute(), "datasets")
    eia_df = pd.read_csv(datasets_dir + "/eia_electricity_hourly.csv")
    eia_df = eia_df.drop(eia_df.columns[0], axis=1)
    eia_subset = pd.DataFrame()
    for _, group in eia_df.groupby("ID"):
        aux = group.iloc[-20000:, :].copy(deep=True)
        eia_subset = pd.concat((eia_subset, aux))
    return eia_subset
