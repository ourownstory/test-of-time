from dataclasses import dataclass
from typing import Tuple

import pandas as pd


def _pivot(df, col_name):
    return df.pivot(index="ds", columns="ID", values=col_name).rename_axis(columns=None).reset_index()


def _melt(df, IDs, col_name):
    return pd.melt(df, id_vars="ds", value_vars=IDs, var_name="ID", value_name=col_name)


SCALING_LEVELS = ["per_dataset", "per_time_series"]


@dataclass
class Scaler:
    """
    A scaling module allowing to perform transform and inverse_transform operations on the time series data. Supports
    transformers from `sklearn.preprocessing` package and other scalers implementing `fit`, `transform` and
    `inverse_transform` methods.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = Scaler(transformer=StandardScaler)
    >>> df_train, df_test = scaler.transform(df_train, df_test)
    >>> fcst_train, fcst_train = scaler.inverse_transform(fcst_train, fcst_train)
    """

    transformer: object
    scaling_level: str

    def __post_init__(self):
        is_transformer_valid = (
            callable(getattr(self.transformer, "fit", None))
            and callable(getattr(self.transformer, "transform", None))
            and callable(getattr(self.transformer, "inverse_transform", None))
        )
        if not is_transformer_valid:
            raise ValueError(
                "Transformer provided to the Scaler must implement fit, transform and inverse_transform methods"
            )

        if self.scaling_level not in SCALING_LEVELS:
            raise ValueError("Invalid scaling level. Allowed levels: `per_dataset`, `per_time_series`")

    def _scale_per_series(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Applies `transform` per series. Fits the `transformer` if `fit` set to True. First, pivot is performed on the
        dataframe so that unique `ID`s become columns, then transformation is applied. Data is returned in the
        original format.

        Parameters:
        -----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
        fit : bool
            if set to True Scaler is fitted with data from `df`

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with transformed data
        """
        IDs = df["ID"].unique()
        df_pivot = _pivot(df, "y")

        df_pivot = self.scale(df_pivot, fit)

        return _melt(df_pivot, IDs, "y")

    def _scale(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """
        Applies `transform` on `y` column in `df`. Fits the `transformer` if `fit` set to True.

        Parameters:
        -----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
        fit : bool
            if set to True Scaler is fitted with data from `df`

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with transformed data
        """
        if fit:
            self.transformer.fit(df["y"])
        df["y"] = self.transformer.transform(df["y"])
        return df

    def _rescale_per_series(self, df: pd.DataFrame, col_name: str) -> pd.Dataframe:
        """
        Applies `inverse_transform` per series. First, pivot is performed on the dataframe so that unique `ID`s
        become columns, then inverse transformation is applied. Operation is repeated on all `yhat` values and
        results are joined to the main df.

        Parameters:
        -----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
        col_name : str
            name of the column, on which the operation is applied

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with scaled data
        """
        IDs = df["ID"].unique()
        df_pivot = _pivot(df, col_name)

        df_pivot = self._rescale(df_pivot, col_name)

        return _melt(df_pivot, IDs, col_name)

    def _rescale(self, df: pd.DataFrame, col_name: str):
        """
        Applies `inverse_transform` on column `col_name` in `df`.

        Parameters:
        -----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
        col_name : str
            name of the column, on which the operation is applied

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with rescaled data
        """
        df[col_name] = self.transformer.inverse_transform(df[col_name])
        return df

    def _inverse_transform(self, df: pd.DataFrame):
        """
        Applies `inverse_transform` per series. First, pivot is performed on the dataframe so that unique `ID`s
        become columns, then inverse transformation is applied. Operation is repeated on all `yhat` values and
        results are joined to the main df.

        Parameters:
        -----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with rescaled data
        """
        if self.scaling_level == "per_time_series":
            apply = self._rescale_per_series
        else:
            apply = self._rescale

        result = apply(df, "y")

        yhats = [col for col in df.columns if "yhat" in col]
        for yhat in yhats:
            result = result.join(apply(df, yhat)[yhat])

        return result

    def transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies `transform` on the dataframes. Scaler is fit on the `df_train`.

        Parameters:
        -----------
        df_train : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with train data
        df_train : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with test data

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with scaled train data
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with scaled test data
        """
        if self.scaling_level == "per_time_series":
            return self._scale_per_series(df_train, fit=True), self._scale_per_series(df_test)

        return self._scale(df_train, fit=True), self._scale(df_test)

    def inverse_transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Applies `inverse_transform` on the dataframes.

        Parameters:
        -----------
        df_train : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with train data
        df_train : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with test data

        Returns:
        --------
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with scaled train data
        pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally ``ID`` with scaled test data
        """
        return self._inverse_transform(df_train), self._inverse_transform(df_test)
