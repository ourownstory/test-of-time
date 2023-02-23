import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Type

import pandas as pd

from tot.df_utils import _check_min_df_len, add_first_inputs_to_df, drop_first_inputs_from_df, prep_or_copy_df
from tot.models.models import Model
from tot.models.utils import _get_seasons, _predict_darts_model, convert_df_to_DartsTimeSeries

log = logging.getLogger("tot.model")

# check import of implemented models and consider order of imports
try:
    from sklearn.linear_model import LinearRegression

    _sklearn_installed = True
except ImportError:
    LinearRegression = None
    _sklearn_installed = False
    raise ImportError(
        "The LinearRegression model could not be imported."
        "Check for proper installation of sklearn: https://scikit-learn.org/stable/install.html"
    )

try:
    from darts.models import RegressionModel

    _darts_installed = True
except ImportError:
    RegressionModel = None
    _darts_installed = False
    raise ImportError(
        "The RegressionModel could not be imported."
        "Check for proper installation of darts: https://github.com/unit8co/darts/blob/master/INSTALL.md"
    )

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
    model_name: str = "Prophet"
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
        self.model = self.model_class(**model_params)
        if custom_seasonalities is not None:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name="{}_daily".format(str(seasonality)),
                    period=seasonality,
                )
        self.n_forecasts = 1
        self.n_lags = 0
        self.season_length = None

    def fit(self, df: pd.DataFrame, freq: str):
        _check_min_df_len(df=df, min_len=self.n_forecasts)
        if "ID" in df.columns and len(df["ID"].unique()) > 1:
            raise NotImplementedError("Prophet does not work with many ts df")
        self.freq = freq
        self.model = self.model.fit(df=df)

    def predict(self, df: pd.DataFrame, df_historic: pd.DataFrame = None):
        _check_min_df_len(df=df, min_len=self.n_forecasts)
        fcst = self.model.predict(df=df)
        fcst_df = pd.DataFrame({"ds": fcst.ds, "y": df.y, "yhat1": fcst.yhat})
        return fcst_df


@dataclass
class LinearRegressionModel(Model):
    """
     A forecasting model using a linear regression of  the target series' lags to obtain a forecast.

     Parameters
     ----------
         n_lags : int
             Previous time series steps to include in auto-regression. Aka AR-order
         output_chunk_length : int
             Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast
             horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may
             be useful if the covariates don't extend far enough into the future.
         model : Type
             Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
             support multi-output regression for multivariate timeseries, in which case one regressor
             will be used per component in the multivariate series.
             If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
         multi_models : bool
             If True, a separate model will be trained for each future lag to predict. If False, a single model is
             trained to predict at step 'output_chunk_length' in the future. Default: True.

     Examples
     --------
     >>> model_classes_and_params = [
     >>>     (
     >>>         LinearRegressionModel,
     >>>         {"lags": 12, "output_chunk_length": 4, "n_forecasts": 4},
     >>>     ),
     >>> ]
     >>>
     >>> benchmark = SimpleBenchmark(
     >>>     model_classes_and_params=model_classes_and_params,
     >>>     datasets=dataset_list,
     >>>     metrics=list(ERROR_FUNCTIONS.keys()),
     >>>     test_percentage=25,
     >>>     save_dir=SAVE_DIR,
     >>>     num_processes=1,
     >>> )

         Note
    ----
        ``Supported capabilities``
        * univariate time series
        * n_forecats > 1
        * autoregression


        ``Not supported capabilities``
        * multivariate time series input
        * frequency check and optional frequency conversion
    """

    model_name: str = "LinearRegressionModel"
    model_class: Type = RegressionModel

    def __post_init__(self):
        # check if installed
        if not (_darts_installed or _sklearn_installed):
            raise RuntimeError(
                "Requires darts and sklearn to be installed:"
                "https://scikit-learn.org/stable/install.html"
                "https://github.com/unit8co/darts/blob/master/INSTALL.md"
            )

        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        model_params.pop("n_forecasts")
        model = LinearRegression(n_jobs=-1)  # n_jobs=-1 indicates to use all processors
        model_params.update({"model": model})  # assign model
        self.model = self.model_class(**model_params)
        self.n_forecasts = self.params["n_forecasts"]
        self.n_lags = model_params["lags"]
        # input checks are provided by model itself

    def fit(self, df: pd.DataFrame, freq: str):
        """Fits the regression model.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with all data
            freq : str
                frequency of the input data
        """
        _check_min_df_len(df=df, min_len=self.n_forecasts + self.n_lags)
        self.freq = freq
        series = convert_df_to_DartsTimeSeries(df, value_cols=df.columns.values[1:-1].tolist(), freq=self.freq)
        self.model = self.model.fit(series)

    def predict(self, df: pd.DataFrame, df_historic: pd.DataFrame = None):
        """Runs the model to make predictions.

        Expects all data to be present in dataframe.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
            df_historic : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally ``ID`` with historic data

        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``, optionally [``ID``], and [``yhat<i>``] where yhat<i> refers to the
                i-step-ahead prediction for this row's datetime, e.g. yhat3 is the prediction for this datetime,
                predicted 3 steps ago, "3 steps old".
        """
        _check_min_df_len(df=df, min_len=self.n_forecasts)
        if df_historic is not None:
            df = self.maybe_extend_df(df_historic, df)
        df, received_ID_col, received_single_time_series, _ = prep_or_copy_df(df)
        fcst_df = _predict_darts_model(
            df=df, model=self, n_req_past_obs=self.n_lags, n_req_future_obs=self.n_forecasts, retrain=False
        )

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
