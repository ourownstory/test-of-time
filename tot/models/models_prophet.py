import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Type

import pandas as pd

from tot.df_utils import _check_min_df_len
from tot.models.models import Model
from tot.models.utils import _get_seasons

log = logging.getLogger("tot.model")

# check import of implemented models and consider order of imports
try:
    from prophet import Prophet

    _prophet_installed = True
except ImportError:
    Prophet = None
    _prophet_installed = False

    raise ImportError(
        "The Prophet model could not be imported."
        "Check for proper installation of prophet: https://facebook.github.io/prophet/docs/installation.html"
    )


@dataclass
class ProphetModel(Model):
    model_class: Type = Prophet

    def __post_init__(self):
        if not _prophet_installed:
            raise RuntimeError("Requires prophet to be installed")
        data_params = self.params["_data_params"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        if "interval_width" in model_params:
            raise NotImplementedError(
                "Quantiles for Prophet not supported in Test-of-Time. Remove interval_width from model input."
            )
        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        self.n_forecasts = 1
        self.n_lags = 0

    def fit(self, df: pd.DataFrame, freq: str, ids_weights: dict):
        """Fits the model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns "ds" and "y" and optionally "ID"
        freq : str
            Frequency of the time series
        ids_weights : dict
            Weights to apply to the loss function per ID

        Returns
        -------
        None
        """
        _check_min_df_len(df=df, min_len=self.n_forecasts)
        if "ID" in df.columns and len(df["ID"].unique()) > 1:
            raise NotImplementedError("Prophet does not work with many ts df")
        self.freq = freq
        self.model = self.model.fit(df=df)

    def predict(self, df: pd.DataFrame, received_single_time_series, df_historic: pd.DataFrame = None):
        """Runs the model to make predictions.

        Expects all data to be present in dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns "ds" and "y" and optionally "ID"
        received_single_time_series : bool
            Whether the df has only one time series
        df_historic : pd.DataFrame
            DataFrame containing column ``ds``, ``y``, and optionally ``ID`` with historic data

        Returns
        -------
        pd.DataFrame
            DataFrame with columns "ds", "y", "yhat1" and "ID"
        """
        _check_min_df_len(df=df, min_len=self.n_forecasts)
        fcst = self.model.predict(df=df)
        fcst_df = pd.DataFrame({"ds": fcst.ds, "y": df.y, "yhat1": fcst.yhat, "ID": df.ID})
        return fcst_df
