import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Type

import pandas as pd
from neuralprophet import NeuralProphet, TorchProphet

from tot.df_utils import (
    _check_min_df_len,
    add_first_inputs_to_df,
    drop_first_inputs_from_df,
    prep_or_copy_df,
    return_df_in_original_format,
)
from tot.models.models import Model
from tot.models.utils import _get_seasons

log = logging.getLogger("tot.model")


@dataclass
class NeuralProphetModel(Model):
    model_name: str = "NeuralProphet"
    model_class: Type = NeuralProphet

    def __post_init__(self):
        data_params = self.params["_data_params"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        if "seasonality_mode" in data_params and data_params["seasonality_mode"] is not None:
            self.params.update({"seasonality_mode": data_params["seasonality_mode"]})
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags
        self.season_length = None

    def fit(self, df: pd.DataFrame, freq: str):
        _check_min_df_len(df=df, min_len=self.n_forecasts + self.n_lags)
        self.freq = freq
        _ = self.model.fit(df=df, freq=freq, progress="none", minimal=True)

    def predict(self, df: pd.DataFrame, df_historic: pd.DataFrame = None):
        _check_min_df_len(df=df, min_len=self.n_forecasts)
        if df_historic is not None:
            df = self.maybe_extend_df(df_historic, df)
        fcst = self.model.predict(df=df)
        (
            fcst,
            received_ID_col,
            received_single_time_series,
            _,
        ) = prep_or_copy_df(fcst)
        fcst_df = pd.DataFrame()
        for df_name, fcst_i in fcst.groupby("ID"):
            y_cols = ["y"] + [col for col in fcst_i.columns if "yhat" in col]
            fcst_aux = pd.DataFrame({"ds": fcst_i.ds})
            for y_col in y_cols:
                fcst_aux[y_col] = fcst_i[y_col]
            fcst_aux["ID"] = df_name
            fcst_df = pd.concat((fcst_df, fcst_aux), ignore_index=True)
        fcst_df = return_df_in_original_format(fcst_df, received_ID_col, received_single_time_series)
        if df_historic is not None:
            fcst_df, df = self.maybe_drop_added_values_from_df(fcst_df, df)
        return fcst_df

    def maybe_extend_df(self, df_train, df_test):
        """
        If model depends on historic values, extend beginning of df_test with last
        df_train values.
        """
        samples = self.n_lags
        df_test = add_first_inputs_to_df(samples=samples, df_train=df_train, df_test=df_test)
        return df_test

    def maybe_drop_added_values_from_df(self, predicted, df):
        """
        If model depends on historic values, drop first values of predicted and df_test.
        """
        samples = self.n_lags
        predicted, df = drop_first_inputs_from_df(samples=samples, predicted=predicted, df=df)
        return predicted, df


@dataclass
class TorchProphetModel(NeuralProphetModel):
    model_name: str = "TorchProphet"
    model_class: Type = TorchProphet

    def __post_init__(self):
        data_params = self.params["_data_params"]
        custom_seasonalities = None
        if "seasonalities" in data_params and len(data_params["seasonalities"]) > 0:
            daily, weekly, yearly, custom_seasonalities = _get_seasons(data_params["seasonalities"])
            self.params.update({"daily_seasonality": daily})
            self.params.update({"weekly_seasonality": weekly})
            self.params.update({"yearly_seasonality": yearly})
        if "seasonality_mode" in data_params and data_params["seasonality_mode"] is not None:
            self.params.update({"seasonality_mode": data_params["seasonality_mode"]})
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        model_params.update({"interval_width": 0})
        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags
        self.season_length = None

    def maybe_extend_df(self, df_train, df_test):
        """
        If model depends on historic values, extend beginning of df_test with last
        df_train values.
        """
        return df_test

    def maybe_drop_added_values_from_df(self, predicted, df):
        """
        If model depends on historic values, drop first values of predicted and df_test.
        """
        return predicted, df
