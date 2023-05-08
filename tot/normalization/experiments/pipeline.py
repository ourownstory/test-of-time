from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from neuralprophet import set_random_seed
from plotly_resampler import unregister_plotly_resampler

from tot.df_utils import (
    _check_min_df_len,
    check_dataframe,
    handle_missing_data,
    prep_or_copy_df,
    return_df_in_original_format,
    split_df,
)
from tot.evaluation.metric_utils import calculate_averaged_metrics_per_experiment

unregister_plotly_resampler()

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


def scale_data_global(df_train: pd.DataFrame, df_test: pd.DataFrame, scaler) -> (pd.DataFrame, pd.DataFrame, dict):
    # initialize the scaler object
    scaler_global = scaler  # e.g. MinMaxScaler(feature_range=(0.1, 1))

    # fit and transform the "y" column
    scaled_y_train = scaler_global.fit_transform(df_train[["y"]])

    # concatenate the scaled "y" values with the original dataframe
    df_train_scaled = pd.concat([df_train[["ds", "ID"]], pd.DataFrame(scaled_y_train, columns=["y"])], axis=1)
    df_train_scaled = df_train_scaled.sort_values(["ID", "ds"])

    scaled_y_test = scaler_global.transform(df_test[["y"]])
    df_test_scaled = pd.concat([df_test[["ds", "ID"]], pd.DataFrame(scaled_y_test, columns=["y"])], axis=1)
    df_test_scaled = df_test_scaled.sort_values(["ID", "ds"])
    scaler_global = {"_df_": scaler_global}

    return df_train_scaled, df_test_scaled, scaler_global


def scale_data_local(df_train: pd.DataFrame, df_test: pd.DataFrame, scaler) -> (pd.DataFrame, pd.DataFrame, dict):
    scalers_local = dict()

    # create an empty dataframe to store the scaled data
    dfs_local_train_scaled = pd.DataFrame()
    dfs_local_test_scaled = pd.DataFrame()

    for id in df_train["ID"].unique():
        # subset the dataframe for the region
        df_train_local = df_train[df_train["ID"] == id].copy().reset_index(drop=True)
        df_test_local = df_test[df_test["ID"] == id].copy().reset_index(drop=True)

        # initialize the scaler object for the region
        scaler_local = scaler

        # fit and transform the "y" column for the region
        scaled_y_train_local = scaler_local.fit_transform(df_train_local[["y"]])

        # concatenate the scaled "y" values with the original dataframe for the region
        df_train_local_scaled = pd.concat(
            [df_train_local[["ds", "ID"]], pd.DataFrame(scaled_y_train_local, columns=["y"]).reset_index(drop=True)],
            axis=1,
        ).reset_index(drop=True)

        # append the scaler object for the region to the list
        scalers_local[id] = scaler_local

        # append the scaled data for the region to the dataframe
        dfs_local_train_scaled = pd.concat([dfs_local_train_scaled, df_train_local_scaled], axis=0)

        scaled_y_test_local = scaler_local.fit_transform(df_test_local[["y"]])

        # concatenate the scaled "y" values with the original dataframe for the region
        df_test_local_scaled = pd.concat(
            [df_test_local[["ds", "ID"]], pd.DataFrame(scaled_y_test_local, columns=["y"]).reset_index(drop=True)],
            axis=1,
        ).reset_index(drop=True)

        # append the scaled data for the region to the dataframe
        dfs_local_test_scaled = pd.concat([dfs_local_test_scaled, df_test_local_scaled], axis=0)

    dfs_local_train_scaled = dfs_local_train_scaled.sort_values(["ID", "ds"])
    dfs_local_test_scaled = dfs_local_test_scaled.sort_values(["ID", "ds"])

    return dfs_local_train_scaled, dfs_local_test_scaled, scalers_local


def scale_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame, scaler=None, scale_level: Optional[str] = "global"
) -> (pd.DataFrame, pd.DataFrame, dict):
    if scaler is None:
        df_train_scaled = df_train
        df_test_scaled = df_test
        scalers = None
    else:
        if scale_level == "global":
            df_train_scaled, df_test_scaled, scalers = scale_data_global(df_train, df_test, scaler)
        elif scale_level == "local":
            df_train_scaled, df_test_scaled, scalers = scale_data_local(df_train, df_test, scaler)
        else:
            raise ValueError("scale_level must be either 'global' or 'local'")

    return df_train_scaled, df_test_scaled, scalers


def data_specific_preprocessing(df, freq, test_percentage=0.25):
    # prep_or_copy_df() ensures that the df has an "ID" column to be usable in the further process
    df, received_ID_col, received_single_time_series, _ = prep_or_copy_df(df)
    # check_dataframe() performs a basic sanity check on the data
    df = check_dataframe(df, check_y=True)
    # handle_missing_data() imputes missing data
    df = handle_missing_data(df, freq=freq)
    # split_df() splits the data into train and test data
    df_train, df_test = split_df(
        df=df,
        test_percentage=test_percentage,
    )
    return df_train, df_test, received_ID_col, received_single_time_series


def model_specific_preprocessing(df_train, df_test, model):
    # check if train and test df contain enough samples
    _check_min_df_len(df=df_train, min_len=model.n_forecasts + model.n_lags)
    _check_min_df_len(df=df_test, min_len=model.n_forecasts)
    # extend the test df with historic values from the train df
    df_test = model.maybe_extend_df(df_train, df_test)
    return df_train, df_test


def fit_and_predict(model, df_train, df_test, freq, received_single_time_series):
    model.fit(df=df_train, freq=freq)
    # the model-individual predict function outputs the forecasts as a df
    fcst_train = model.predict(df=df_train, received_single_time_series=received_single_time_series)
    fcst_test = model.predict(df=df_test, received_single_time_series=received_single_time_series)
    return fcst_train, fcst_test


def model_specific_postprocessing(fcst_test, df_test, model):
    # As you can see, the method is a class method and hence linked to the model
    fcst_test = model.maybe_drop_added_values_from_df(predicted=fcst_test, df=df_test)
    return fcst_test


def data_specific_postprocessing(fcst_train, fcst_test, received_ID_col, received_single_time_series):
    # in case an 'ID' column was previously added, return_df_in_original_format() will remove it again
    fcst_train_df = return_df_in_original_format(fcst_train, received_ID_col, received_single_time_series)
    fcst_test_df = return_df_in_original_format(fcst_test, received_ID_col, received_single_time_series)
    # in case, missing data was imputed maybe_drop_added_dates()
    return fcst_train_df, fcst_test_df


def rescale_data_global(
    fcst_train: pd.DataFrame, fcst_test: pd.DataFrame, fitted_scalers: dict
) -> (pd.DataFrame, pd.DataFrame):
    yhat = [col for col in fcst_train.columns if "yhat" in col]
    yhat.append("y")
    # fcst_train_yhat_rescaled = fcst_train[yhat].apply(lambda x: fitted_scaler[0].inverse_transform([x]))
    yhat_rescaled_train = fitted_scalers["_df_"].inverse_transform(fcst_train[yhat])
    fcst_train_yhat_rescaled = pd.DataFrame(yhat_rescaled_train, index=fcst_train.index, columns=yhat)
    fcst_train_rescaled = fcst_train.copy(deep=True)
    fcst_train_rescaled[yhat] = fcst_train_yhat_rescaled[yhat].reset_index(drop=True)
    fcst_train_rescaled = fcst_train_rescaled.sort_values(["ID", "ds"])

    yhat_rescaled_test = fitted_scalers["_df_"].inverse_transform(fcst_test[yhat])
    fcst_test_yhat_rescaled = pd.DataFrame(yhat_rescaled_test, index=fcst_test.index, columns=yhat)
    fcst_test_rescaled = fcst_test.copy(deep=True)
    fcst_test_rescaled[yhat] = fcst_test_yhat_rescaled[yhat].reset_index(drop=True)
    fcst_test_rescaled = fcst_test_rescaled.sort_values(["ID", "ds"])
    return fcst_train_rescaled, fcst_test_rescaled


def rescale_data_local(
    fcst_train: pd.DataFrame, fcst_test: pd.DataFrame, fitted_scalers: dict
) -> (pd.DataFrame, pd.DataFrame):
    rescaled_dfs_test_local = pd.DataFrame()
    rescaled_dfs_train_local = pd.DataFrame()

    yhat = [col for col in fcst_train.columns if "yhat" in col]
    yhat.append("y")

    for id in fcst_train["ID"].unique():
        # subset the dataframe for the region
        fcst_train_local = fcst_train[fcst_train["ID"] == id]
        fcst_test_local = fcst_test[fcst_test["ID"] == id]
        #
        # get the corresponding scaler object for the region
        scaler_local = fitted_scalers[id]

        rescaled_train_local = scaler_local.inverse_transform(fcst_train_local[yhat])
        # create a new dataframe with the retransformed "y" values for the region
        fcst_train_yhat_rescaled_local = pd.DataFrame(rescaled_train_local, index=fcst_train_local.index, columns=yhat)
        fcst_train_rescaled_local = fcst_train_local.copy(deep=True)
        fcst_train_rescaled_local[yhat] = fcst_train_yhat_rescaled_local[yhat]
        fcst_train_rescaled_local = fcst_train_rescaled_local.sort_values(["ID", "ds"])

        rescaled_test_local = scaler_local.inverse_transform(fcst_test_local[yhat])
        # create a new dataframe with the retransformed "y" values for the region
        fcst_test_yhat_rescaled_local = pd.DataFrame(rescaled_test_local, index=fcst_test_local.index, columns=yhat)
        fcst_test_rescaled_local = fcst_test_local.copy(deep=True)
        fcst_test_rescaled_local[yhat] = fcst_test_yhat_rescaled_local[yhat]
        fcst_test_rescaled_local = fcst_test_rescaled_local.sort_values(["ID", "ds"])

        # append the retransformed data for the region to the new dataframe
        rescaled_dfs_train_local = pd.concat([rescaled_dfs_train_local, fcst_train_rescaled_local])
        rescaled_dfs_test_local = pd.concat([rescaled_dfs_test_local, fcst_test_rescaled_local])

    rescaled_dfs_train_local = rescaled_dfs_train_local.sort_values(["ID", "ds"])
    rescaled_dfs_test_local = rescaled_dfs_test_local.sort_values(["ID", "ds"])

    return rescaled_dfs_train_local, rescaled_dfs_test_local


def rescale_data(
    fcst_train: pd.DataFrame, fcst_test: pd.DataFrame, fitted_scalers: dict, scale_level: Optional[str] = "global"
) -> (pd.DataFrame, pd.DataFrame):
    if fitted_scalers is None:
        return fcst_train, fcst_test
    else:
        if scale_level == "global":
            df_train_rescaled, df_test_rescaled = rescale_data_global(fcst_train, fcst_test, fitted_scalers)
        elif scale_level == "local":
            df_train_rescaled, df_test_rescaled = rescale_data_local(fcst_train, fcst_test, fitted_scalers)
        else:
            raise ValueError("scale_level must be either 'global' or 'local'")

        return df_train_rescaled, df_test_rescaled


def run_custom_pipline(df, model_class, params, freq, metrics, scaler=None, test_percentage=0.25, scale_level=None):
    # Data-specific pre-processing
    df_train, df_test, received_ID_col, received_single_time_series = data_specific_preprocessing(
        df, freq, test_percentage
    )

    df_train, df_test, fitted_scalers = scale_data(df_train, df_test, scaler=scaler, scale_level=scale_level)

    set_random_seed(42)
    model = model_class(params=params)
    # Model-specific data pre-processing
    df_train, df_test = model_specific_preprocessing(df_train, df_test, model)
    # Fit and predict model
    fcst_train, fcst_test = fit_and_predict(
        model=model,
        df_train=df_train,
        df_test=df_test,
        freq=freq,
        received_single_time_series=received_single_time_series,
    )
    # Model-specific data post-processing

    fcst_test = model_specific_postprocessing(fcst_test=fcst_test, df_test=df_test, model=model)
    # Data-specific data post-processing
    fcst_train, fcst_test = data_specific_postprocessing(
        fcst_train,
        fcst_test,
        received_ID_col=True,  # leave ID col in --> needs to be fixed
        received_single_time_series=received_single_time_series,
    )
    # Data re-scaling
    fcst_train, fcst_test = rescale_data(fcst_train, fcst_test, fitted_scalers=fitted_scalers, scale_level=scale_level)

    # Evaluation
    result_train = calculate_averaged_metrics_per_experiment(
        fcst_df=fcst_train, df_historic=fcst_train, metrics=metrics, metadata=None, freq=freq
    )
    result_test = calculate_averaged_metrics_per_experiment(
        fcst_df=fcst_test, df_historic=fcst_train, metrics=metrics, metadata=None, freq=freq
    )

    return result_train, result_test, fcst_train, fcst_test


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
    fig.update_layout(layout)
    return fig
