import logging

import arrow
import numpy as np
import plotly.graph_objs as go
from plotly_resampler import register_plotly_resampler, unregister_plotly_resampler

from tot.df_utils import prep_or_copy_df
from tot.error_utils import raise_if

log = logging.getLogger("tot.plot")

# UI Configuration
prediction_color = "#2d92ff"
actual_color = "black"
trend_color = "#B23B00"
line_width = 2
marker_size = 4
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
layout_args = {
    "autosize": True,
    "template": "plotly_white",
    "margin": go.layout.Margin(l=0, r=10, b=0, t=10, pad=0),
    "font": dict(size=10),
    "title": dict(font=dict(size=12)),
    "hovermode": "x unified",
}


def is_jupyter_notebook():
    """
    Determine if the code is being executed in a Jupyter notebook environment.

    Returns
    -------
    bool :
        True if the code is being executed in a Jupyter notebook, False otherwise.
    """
    from IPython import get_ipython

    if "google.colab" in str(get_ipython()):
        return False

    try:
        from IPython.core.getipython import get_ipython

        return "ipykernel" in str(get_ipython())  # pragma: no cover

    except [ImportError, AttributeError]:
        return False


def select_plotting_backend(plotting_backend):
    """
    Automatically select the plotting backend.

    Given `plotting_backend`, returns "plotly-resampler" if `validate_current_env_for_resampler()`
    returns `True` and `plotting_backend` is None, "plotly" otherwise. If
    `plotting_backend` is not None, returns `plotting_backend.lower()`.

    Parameters
    ----------
    plotting_backend: str
        plotting backend given by the user.

    Returns
    -------
    str
        The new plotting backend.
    """
    if plotting_backend is None:
        if is_jupyter_notebook():
            plotting_backend = "plotly-resampler"
        else:
            log.warning(
                "Warning: plotly-resampler not supported for the environment you are using. "
                "Plotting backend is set to 'plotly' without resampling "
            )
            plotting_backend = "plotly"
    elif plotting_backend == "plotly-resampler":
        if not is_jupyter_notebook():
            log.warning(
                "Warning: plotly-resampler not supported for the environment you are using. "
                "Consider switching plotting_backend to 'plotly' or 'matplotlib "
            )
    return plotting_backend.lower()


def validate_plotting_backend_input(plotting_backend):
    """
    Validate the input argument for the plotting backend.

    Parameters
    ----------
    plotting_backend:str
        The name of the plotting backend.

    Raises
    ----------
    ValueError:
        If the plotting backend is not a valid backend.

    Returns
    ----------
        None
    """
    valid_plotting_backends = [None, "plotly", "plotly-resampler"]
    raise_if(
        plotting_backend not in valid_plotting_backends,
        "Selected plotting backend invalid. Set plotting backend to one of the "
        "valid options 'plotly','plotly-auto','plotly-resampler'.",
    )


def validate_highlight_forecast_input(highlight_forecast, fcst):
    """
    Validate the input argument for the highlight_forecast.

    Parameters
    ----------
    highlight_forecast : int
        The number of forecasts to highlight.
    fcst : pd.DataFrame
        The forecast DataFrame.

    Raises
    ------
    ValueError
        If the highlight_forecast value is greater than the number of yhat (prediction horizon) columns in fcst.

    Returns
    -------
    None
    """
    n_yhat = len([col for col in fcst.columns if "yhat" in col])
    is_highlight_forecast_valid = highlight_forecast is None or highlight_forecast < n_yhat
    raise_if(
        not is_highlight_forecast_valid,
        "Input for highlight_forecast invalid. "
        "Set highlight_forecast step equal to "
        " or smaller than the prediction horizon",
    )


def validate_df_name_input(df_name, fcst):
    """
    Validate the input df_name and returns a dataframe with a single time series and an ID.

    Parameters
    ----------
    df_name : str
        optoinal, ID from time series that should be plotted
    fcst : pd.DataFrame
        forecast dataframe

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe containing the time series data for the specified name.

    Raises
    ------
    ValueError
        If the input DataFrame contains more than one time series and the df_name argument is not provided, or if the
        specified df_name is not present in the DataFrame.

    """
    fcst, received_ID_col, received_single_time_series, _ = prep_or_copy_df(fcst)
    if not received_single_time_series:
        if df_name not in fcst["ID"].unique():
            raise_if(
                len(fcst["ID"].unique()) > 1,
                "Many time series are present in the pd.DataFrame (more than one "
                "ID). Please, "
                "especify ID to be plotted.",
            )
        fcst = fcst[fcst["ID"] == df_name].copy(deep=True)
        log.info(f"Plotting data from ID {df_name}")
    return fcst


def _plot_plotly(
    fcst,
    quantiles=[0.5],
    xlabel="ds",
    ylabel="y",
    highlight_forecast=None,
    figsize=(700, 350),
    resampler_active=False,
):
    """
    Plot the NeuralProphet forecast

    Parameters
    ---------
        fcst : pd.DataFrame
            Output of m.predict
        quantiles: list
            Quantiles for which the forecasts are to be plotted.
        xlabel : str
            Label name on X-axis
        ylabel : str
            Label name on Y-axis
        highlight_forecast : Union[int, None]
            i-th step ahead forecast to highlight.
        line_per_origin : bool
            Print a line per forecast of one per forecast age
        figsize : tuple [int, int]
            Width, height in inches.

    Returns
    -------
        Plotly figure
    """
    if resampler_active:
        unregister_plotly_resampler()

        # register_plotly_resampler(mode="auto")
    else:
        unregister_plotly_resampler()
    cross_marker_color = "blue"
    cross_symbol = "x"

    fcst = fcst.fillna(value=np.nan)

    ds = fcst["ds"].apply(lambda x: arrow.get(x).datetime)
    colname = "yhat"
    step = 1
    # all yhat column names, including quantiles
    yhat_col_names = [col_name for col_name in fcst.columns if f"{colname}" in col_name]
    # without quants
    yhat_col_names_no_qts = [
        col_name for col_name in yhat_col_names if f"{colname}" in col_name and "%" not in col_name
    ]
    data = []

    if highlight_forecast is None:
        for i, yhat_col_name in enumerate(yhat_col_names_no_qts):
            data.append(
                go.Scatter(
                    name=yhat_col_name,
                    x=ds,
                    y=fcst[f"{colname}{i + 1}"],
                    mode="lines",
                    line=dict(color=f"rgba(45, 146, 255, {0.2 + 2.0 / (i + 2.5)})", width=line_width),
                    fill="none",
                )
            )
    if len(quantiles) > 1:
        for i in range(1, len(quantiles)):
            # skip fill="tonexty" for the first quantile
            data.append(
                go.Scatter(
                    name=f"{colname}{highlight_forecast if highlight_forecast else step} {round(quantiles[i] * 100, 1)}%",
                    x=ds,
                    y=fcst[
                        f"{colname}{highlight_forecast if highlight_forecast else step} {round(quantiles[i] * 100, 1)}%"
                    ],
                    mode="lines",
                    line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                    fill="none" if i == 1 else "tonexty",
                    fillcolor="rgba(45, 146, 255, 0.2)",
                )
            )

    if highlight_forecast is not None:
        x = ds
        y = fcst[f"yhat{highlight_forecast}"]
        data.append(
            go.Scatter(
                name="Predicted",
                x=x,
                y=y,
                mode="lines",
                line=dict(color=prediction_color, width=line_width),
            )
        )
        data.append(
            go.Scatter(
                name="Predicted",
                x=x,
                y=y,
                mode="markers",
                marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
            )
        )

    # Add actual
    data.append(
        go.Scatter(name="Actual", x=ds, y=fcst["y"], marker=dict(color=actual_color, size=marker_size), mode="markers")
    )

    layout = go.Layout(
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        xaxis=go.layout.XAxis(
            title=xlabel,
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
        yaxis=go.layout.YAxis(title=ylabel, **yaxis_args),
        **layout_args,
    )
    fig = go.Figure(data=data, layout=layout)
    unregister_plotly_resampler()
    return fig
