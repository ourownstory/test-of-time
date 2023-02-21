import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger("tot.df_utils")


def convert_to_datetime(series: pd.Series) -> pd.Series:
    """Convert input series to datetime format

    Parameters
    ----------
        series : pd.Series
            input series that needs to be converted to datetime format

    Returns
    -------
        pd.Series
            series in datetime format

    Raises
    ------
        ValueError
            if input series contains NaN values or has timezone specified
    """
    if series.isnull().any():
        raise ValueError("Found NaN in column ds.")
    if series.dtype == np.int64:
        series = series.astype(str)
    if not np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series)
    if series.dt.tz is not None:
        raise ValueError("Column ds has timezone specified, which is not supported. Remove timezone.")
    return series


def _split_df(df: pd.DataFrame, test_percentage: Union[float, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a timeseries DataFrame into train and validation sets.
    The function expects the DataFrame to have a single ID column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be split.
    test_percentage : float, int
        The percentage or number of samples to be used for validation.
        If the value is between 0 and 1, it is interpreted as a percentage of the total number of samples.
        If the value is greater than or equal to 1, it is interpreted as the number of samples to be used for validation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training DataFrame and the validation DataFrame.
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    n_samples = len(df)
    if 0.0 < test_percentage < 1.0:
        n_valid = max(1, int(n_samples * test_percentage))
    else:
        assert test_percentage >= 1
        assert type(test_percentage) == int
        n_valid = test_percentage
    n_train = n_samples - n_valid
    assert n_train >= 1

    split_idx_train = n_train
    split_idx_val = split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    log.debug(f"{n_train} n_train, {n_samples - n_train} n_eval")
    return df_train, df_val


def split_df(
    df: pd.DataFrame, test_percentage: Union[float, int] = 0.25, local_split: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits timeseries dataframe into train and validation sets.
        Parameters:
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            test_percentage : (Union[float, int])
               fraction (0,1) of data to use for holdout validation set, or number of validation samples >1
            local_split : bool
                when set to true, each time series of a dataframes will be split locally
                (default): True

        Returns
        -------
            Tuple[pd.DataFrame, pd.DataFrame]
                training data as pd.DataFrame and validation data as pd.DataFrame
    """
    df, _, _, _ = prep_or_copy_df(df)
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    if local_split:
        for df_name, df_i in df.groupby("ID"):
            df_t, df_v = _split_df(df_i, test_percentage)
            df_train = pd.concat((df_train, df_t.copy(deep=True)), ignore_index=True)
            df_val = pd.concat((df_val, df_v.copy(deep=True)), ignore_index=True)
    else:
        if len(df["ID"].unique()) == 1:
            for df_name, df_i in df.groupby("ID"):
                df_train, df_val = _split_df(df_i, test_percentage)
        # TODO: provide case for multiple time series and split by time threshold
    # df_train and df_val are returned as pd.DataFrames
    return df_train, df_val


def __crossvalidation_split_df(df, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            data
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds (default: 0.0)

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    total_samples = len(df)
    samples_fold = max(1, int(fold_pct * total_samples))
    samples_overlap = int(fold_overlap_pct * samples_fold)
    assert samples_overlap < samples_fold
    min_train = total_samples - samples_fold - (k - 1) * (samples_fold - samples_overlap)
    assert (
        min_train >= samples_fold
    ), "Test percentage too large. Not enough train samples. Select smaller test percentage. "
    folds = []
    df_fold = df.copy(deep=True)
    for i in range(k, 0, -1):
        df_train, df_val = split_df(df_fold, test_percentage=samples_fold)
        folds.append((df_train, df_val))
        split_idx = len(df_fold) - samples_fold + samples_overlap
        df_fold = df_fold.iloc[:split_idx].reset_index(drop=True)
    folds = folds[::-1]
    return folds


def _crossvalidation_split_df(df, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            data
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds (default: 0.0)
        global_model_cv_type : str
            Type of crossvalidation to apply to the time series.

                options:

                    ``global-time`` (default) crossvalidation is performed according to a time stamp threshold.

                    ``local`` each episode will be crossvalidated locally (may cause time leakage among different episodes)

                    ``intersect`` only the time intersection of all the episodes will be considered. A considerable amount of data may not be used. However, this approach guarantees an equal number of train/test samples for each episode.

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
    """
    if len(df["ID"].unique()) == 1:
        for df_name, df_i in df.groupby("ID"):
            folds = __crossvalidation_split_df(df_i, k, fold_pct, fold_overlap_pct)
    else:
        # implement procedure for multiple IDs
        pass

    return folds


def crossvalidation_split_df(df, freq, k=5, fold_pct=0.1, fold_overlap_pct=0.5):
    """Splits timeseries data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        freq : str
            data step sizes. Frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds.

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data

    See Also
    --------
        split_df : Splits timeseries df into train and validation sets.

    Examples
    --------
        >>> df1 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-01', periods = 10, freq = 'D'),
        ...                     'y': [9.59, 8.52, 8.18, 8.07, 7.89, 8.09, 7.84, 7.65, 8.71, 8.09]})
        >>> df1
            ds	        y
        0	2022-12-03	7.67
        1	2022-12-04	7.64
        2	2022-12-05	7.55
        3	2022-12-06	8.25
        4	2022-12-07	8.32
        5	2022-12-08	9.59
        6	2022-12-09	8.52
        7	2022-12-10	7.55
        8	2022-12-11	8.25
        9	2022-12-12	8.09
    """
    df, received_ID_col, received_single_time_series, _ = prep_or_copy_df(df)
    # df = self._check_dataframe(df, check_y=False, exogenous=False) #TODO: add via restructured pipeline
    # freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq) #TODO: add via restructured pipeline
    # df = model._handle_missing_data(df, freq=freq, predicting=False) #TODO: add via restructured pipeline
    folds = _crossvalidation_split_df(
        df,
        k=k,
        fold_pct=fold_pct,
        fold_overlap_pct=fold_overlap_pct,
    )
    # if not received_ID_col and received_single_time_series:
    #     # Delete ID column (__df__) of df_train and df_val of all folds in case ID was not previously provided
    #     new_folds = []
    #     for i in range(len(folds)):
    #         df_train = return_df_in_original_format(folds[i][0])
    #         df_val = return_df_in_original_format(folds[i][1])
    #         new_folds.append((df_train, df_val))
    #     folds = new_folds
    return folds


def _check_min_df_len(df, min_len):
    """
    Check that the provided dataframe has a minimum number of rows.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to check the length of
    min_len : int
        minimum number of rows required for the dataframe

    Raises
    ------
    AssertionError
        If the dataframe does not have at least `min_len` rows.
    """
    assert len(df) > min_len, "df has not enough data to create a single input sample."
    # TODO: adapt for multi time series df


def add_first_inputs_to_df(samples: int, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Add the last `samples` of data from df_train to the start of df_test.

    Parameters
    ----------
    samples: int
        Number of last samples from df_train to be added to start of df_test
    df_train: pd.DataFrame
        Dataframe containing training data
    df_test: pd.DataFrame
        Dataframe containing testing data

    Returns
    -------
    df_test: pd.DataFrame
        Dataframe containing testing data with the last `samples` of data from df_train at the start.
    """
    df_test_new = pd.DataFrame()

    for df_name, df_test_i in df_test.groupby("ID"):
        df_train_i = df_train[df_train["ID"] == df_name].copy(deep=True)
        df_test_i = pd.concat(
            [df_train_i.tail(samples), df_test_i],
            ignore_index=True,
        )
        df_test_new = pd.concat((df_test_new, df_test_i), ignore_index=True)
    return df_test_new


def drop_first_inputs_from_df(
    samples: int, predicted: pd.DataFrame, df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drops the first 'samples' number of rows from both 'predicted' and 'df' dataframes for each group of 'ID' column.

    Parameters:
        samples : int
            Number of rows to be dropped from the start of each group of 'ID' column in 'predicted' and 'df' dataframes.
        predicted : pd.DataFrame
            Dataframe containing the predicted values.
        df : pd.DataFrame
            Dataframe containing the actual values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing the modified 'predicted' and 'df' dataframes.
    """
    predicted_new = pd.DataFrame()
    df_new = pd.DataFrame()

    for df_name, df_i in df.groupby("ID"):
        predicted_i = predicted[predicted["ID"] == df_name].copy(deep=True)
        predicted_i = predicted_i[samples:]
        df_i = df_i[samples:]
        df_new = pd.concat((df_new, df_i), ignore_index=True)
        predicted_new = pd.concat((predicted_new, predicted_i), ignore_index=True)
    return predicted_new, df_new


def maybe_drop_added_dates(predicted: pd.DataFrame, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes any dates in predicted which are not in df_test, if model imputed any dates.

    Parameters
    ----------
    predicted: pd.DataFrame
        Dataframe containing the predicted values
    df: pd.DataFrame
        Dataframe containing the test values

    Returns
    -------
    predicted: pd.DataFrame
        Dataframe containing the predicted values with any added dates removed
    df: pd.DataFrame
        Dataframe containing the test values
    """
    predicted_new = pd.DataFrame()
    df_new = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        predicted_i = predicted[predicted["ID"] == df_name].copy(deep=True)
        predicted_i["ds"] = convert_to_datetime(predicted_i["ds"])
        df_i["ds"] = convert_to_datetime(df_i["ds"])
        df_i.set_index("ds", inplace=True)
        predicted_i.set_index("ds", inplace=True)
        predicted_i = predicted_i.loc[df_i.index]
        predicted_i = predicted_i.reset_index()
        df_i = df_i.reset_index()
        df_new = pd.concat((df_new, df_i), ignore_index=True)
        predicted_new = pd.concat((predicted_new, predicted_i), ignore_index=True)
    return predicted, df


def _add_missing_dates_nan(df, freq):
    """Fills missing datetimes in ``ds``, with NaN for all other columns

    Parameters
    ----------
        df : pd.Dataframe
            with column ``ds``  datetimes
        freq : str
            Frequency of data recording, any valid frequency for pd.date_range,
            such as ``D`` or ``M``

    Returns
    -------
        pd.DataFrame
            dataframe without date-gaps but nan-values
    """
    data_len = len(df)
    r = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq=freq)
    df_all = df.set_index("ds").reindex(r).rename_axis("ds").reset_index()
    num_added = len(df_all) - data_len
    return df_all, num_added


def _fill_linear_then_rolling_avg(series, limit_linear, rolling):
    """Adds missing dates, fills missing values with linear imputation or trend.

    Parameters
    ----------
        series : pd.Series
            series with nan to be filled in.
        limit_linear : int
            maximum number of missing values to impute.

            Note
            ----
            because imputation is done in both directions, this value is effectively doubled.

        rolling : int
            maximal number of missing values to impute.

            Note
            ----
            window width is rolling + 2*limit_linear

    Returns
    -------
        pd.DataFrame
            manipulated dataframe containing filled values
    """
    # impute small gaps linearly:
    series = pd.to_numeric(series)
    series = series.interpolate(method="linear", limit=limit_linear, limit_direction="both")
    # fill remaining gaps with rolling avg
    is_na = pd.isna(series)
    rolling_avg = series.rolling(rolling + 2 * limit_linear, min_periods=2 * limit_linear, center=True).mean()
    series.loc[is_na] = rolling_avg[is_na]
    remaining_na = sum(series.isnull())
    return series, remaining_na


def _handle_missing_data(df, freq):
    """Checks and normalizes new data

    Data is also auto-imputed, since impute_missing is manually set to ``True``.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y`` with all data
        freq : str
            data step sizes. Frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.

    Returns
    -------
        pd.DataFrame
            preprocessed dataframe
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1

    # set imput parameters:
    impute_linear = 10
    impute_rolling = 10

    impute_missing = True
    # add missing dates
    df, missing_dates = _add_missing_dates_nan(df, freq=freq)
    if missing_dates > 0:
        if impute_missing:
            log.info(f"{missing_dates} missing dates added.")

    nan_at_end = 0
    while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
        nan_at_end += 1
    if nan_at_end > 0:
        # training - drop nans at end
        df = df[:-nan_at_end]
        log.info(
            f"Dropped {nan_at_end} consecutive nans at end. "
            "Training data can only be imputed up to last observation."
        )
    # TODO: add regressors

    # impute missing values
    data_columns = []
    data_columns.append("y")
    for column in data_columns:
        sum_na = sum(df[column].isnull())
        if sum_na > 0:
            log.warning(f"{sum_na} missing values in column {column} were detected in total. ")
            if impute_missing:
                # else:
                df.loc[:, column], remaining_na = _fill_linear_then_rolling_avg(
                    df[column],
                    limit_linear=impute_linear,  # TODO: store in config
                    rolling=impute_rolling,  # TODO: store in config
                )
                log.info(f"{sum_na - remaining_na} NaN values in column {column} were auto-imputed.")
                if remaining_na > 0:
                    log.warning(
                        f"More than {2 * impute_linear + impute_rolling} consecutive missing values encountered in column {column}. "
                        f"{remaining_na} NA remain after auto-imputation. "
                    )
    return df


def handle_missing_data(df, freq):
    """Checks and normalizes new data

    Data is also auto-imputed, since impute_missing is manually set to ``True``.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        freq : str
            data step sizes. Frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.

    Returns
    -------
        pre-processed df
    """
    df, _, _, _ = prep_or_copy_df(df)
    df_handled_missing = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        df_handled_missing_aux = _handle_missing_data(df_i, freq).copy(deep=True)
        df_handled_missing_aux["ID"] = df_name
        df_handled_missing = pd.concat((df_handled_missing, df_handled_missing_aux), ignore_index=True)
    return df_handled_missing


def check_single_dataframe(df, check_y):
    """Performs basic data sanity checks and ordering
    as well as prepare dataframe for fitting or predicting.

    Parameters
    ----------
        df : pd.DataFrame
            with columns ds
        check_y : bool
            if df must have series values (``True`` if training or predicting with autoregression)

    Returns
    -------
        pd.DataFrame
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    if df.shape[0] == 0:
        raise ValueError("Dataframe has no rows.")
    if "ds" not in df:
        raise ValueError('Dataframe must have columns "ds" with the dates.')
    if df.loc[:, "ds"].isnull().any():
        raise ValueError("Found NaN in column ds.")
    if df["ds"].dtype == np.int64:
        df["ds"] = df.loc[:, "ds"].astype(str)
    if pd.api.types.is_string_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df.loc[:, "ds"])
    if not np.issubdtype(df["ds"].dtype, np.datetime64):
        df["ds"] = pd.to_datetime(df.loc[:, "ds"])
    if df["ds"].dt.tz is not None:
        raise ValueError("Column ds has timezone specified, which is not supported. Remove timezone.")
    if len(df.ds.unique()) != len(df.ds):
        raise ValueError("Column ds has duplicate values. Please remove duplicates.")

    columns = []
    if check_y:
        columns.append("y")

    for name in columns:
        if name not in df:
            raise ValueError(f"Column {name!r} missing from dataframe")
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError(f"Dataframe column {name!r} only has NaN rows.")
        if not np.issubdtype(df[name].dtype, np.number):
            df.loc[:, name] = pd.to_numeric(df.loc[:, name])
        if np.isinf(df.loc[:, name].values).any():
            df.loc[:, name] = df[name].replace([np.inf, -np.inf], np.nan)
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError(f"Dataframe column {name!r} only has NaN rows.")

    if df.index.name == "ds":
        df.index.name = None
    df = df.sort_values("ds")
    df = df.reset_index(drop=True)
    return df


def check_dataframe(df, check_y=True):
    """Performs basic data sanity checks and ordering,
    as well as prepare dataframe for fitting or predicting.

    Parameters
    ----------
        df : pd.DataFrame
            containing column ``ds``
        check_y : bool
            if df must have series values
            set to True if training or predicting with autoregression

    Returns
    -------
        pd.DataFrame or dict
            checked dataframe
    """
    checked_df = pd.DataFrame()

    for df_name, df_i in df.groupby("ID"):
        df_aux = check_single_dataframe(df_i, check_y)
        df_aux = df_aux.copy(deep=True)
        df_aux["ID"] = df_name
        checked_df = pd.concat((checked_df, df_aux), ignore_index=True)
    return checked_df


def prep_or_copy_df(df):
    """Copy df if it contains the ID column. Creates ID column with '__df__' if it is a df with a single time series.
    Parameters
    ----------
        df : pd.DataFrame
            df or dict containing data
    Returns
    -------
        pd.DataFrames
            df with ID col
        bool
            whether the ID col was present
        bool
            wheter it is a single time series
    """
    received_ID_col = False
    received_single_time_series = True
    if isinstance(df, pd.DataFrame):
        new_df = df.copy(deep=True)
        if "ID" in df.columns:
            received_ID_col = True
            log.debug("Received df with ID col")
            if len(new_df["ID"].unique()) > 1:
                log.debug("Received df with many time series")
                received_single_time_series = False
            else:
                log.debug("Received df with single time series")
        else:
            new_df["ID"] = "__df__"
            log.debug("Received df with single time series")
    elif df is None:
        raise ValueError("df is None")
    else:
        raise ValueError("Please, insert valid df type (pd.DataFrame)")

    # list of IDs
    id_list = list(new_df.ID.unique())

    return new_df, received_ID_col, received_single_time_series, id_list


def return_df_in_original_format(df, received_ID_col=False, received_single_time_series=True):
    """Return dataframe in the original format.

    Parameters
    ----------
        df : pd.DataFrame
            df with data
        received_ID_col : bool
            whether the ID col was present
        received_single_time_series: bool
            wheter it is a single time series
    Returns
    -------
        pd.Dataframe
            original input format
    """
    new_df = df.copy(deep=True)
    if not received_ID_col and received_single_time_series:
        assert len(new_df["ID"].unique()) == 1
        new_df.drop("ID", axis=1, inplace=True)
        log.info("Returning df with no ID column")
    return new_df
