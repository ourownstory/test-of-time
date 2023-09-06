import os
import random

import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly_resampler import unregister_plotly_resampler

from tot.plotting import plot_plotly

unregister_plotly_resampler()


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


def plot_ts(df, range_start=0):
    fig = go.Figure()

    for group_name, group_data in df.groupby("ID"):
        fig.add_trace(
            go.Scatter(
                x=group_data.iloc[range_start:, group_data.columns.get_loc("ds")],
                y=group_data.iloc[range_start:, group_data.columns.get_loc("y")],
                mode="lines",
                name=group_name,
            )
        )

    fig.update_layout(layout())
    return fig


def plot_and_save(df, plot=True, save=False, file_name=None):
    fig = plot_ts(df)
    fig2 = plot_ts(df, range_start=-24 * 7)
    if plot:
        fig.show()
        fig2.show()
    if save:
        fig.write_image(file_name)
        fig2.write_image(file_name[:-4] + "_zoomed.png")


def plot_forecast(df, plot=True, save=True, file_name=None, file_name_fcst=None):
    unique_ids = df["ID"].unique()
    n_ids = len(unique_ids)
    if n_ids > 10:
        sampled_ids = random.sample(list(unique_ids), k=10)
        n_ids = len(sampled_ids)
    else:
        sampled_ids = unique_ids
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
        df.to_csv(file_name_fcst)


def plot_forecasts(benchmark, dir_name, plot=False, save=True):
    for exp, fcst_test in zip(benchmark.experiments, benchmark.fcst_test):
        plot_file_name = os.path.join(dir_name, f"{exp.experiment_name}.png")
        file_name_fcst = os.path.join(dir_name, f"{exp.experiment_name}.csv")
        plot_forecast(fcst_test, plot=plot, save=save, file_name=plot_file_name, file_name_fcst=file_name_fcst)
