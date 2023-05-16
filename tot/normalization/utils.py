import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def plot(df):
    fig = go.Figure()

    for region in df["ID"].unique():
        fig.add_trace(
            go.Scatter(
                name=region,
                x=df[df['ID'] == region]['ds'],
                y=df[df['ID'] == region]['y']))
    fig.show()


def filter_ID(df, values):
    return df[df.ID.isin(values)]


def plot_forecast(df):
    fig = go.Figure()

    for region in df["ID"].unique():
        fig.add_trace(
            go.Scatter(
                name=region + ' predicted',
                x=df[df['ID'] == region]['ds'],
                y=df[df['ID'] == region]['yhat1']))
        fig.add_trace(
            go.Scatter(
                name=region + ' actual',
                mode="markers",
                x=df[df['ID'] == region]['ds'],
                y=df[df['ID'] == region]['y']))
    fig.show()


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