import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

from tot.error_utils import raise_data_validation_error_if, raise_if

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
    raise_if(series.isnull().any(), "Found NaN in column ds.")
    if series.dtype == np.int64:
        series = series.astype(str)
    if not np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series)
    raise_if(series.dt.tz is not None, "Column ds has timezone specified, which is not supported. Remove timezone.")
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
    _validate_single_ID_df(df)

    n_samples = len(df)
    n_train = _calculate_n_train(n_samples, test_percentage)

    split_idx_train = n_train
    split_idx_val = split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    log.debug(f"{n_train} n_train, {n_samples - n_train} n_eval")
    return df_train, df_val


def _validate_single_ID_df(df: pd.DataFrame) -> None:
    """Check if the DataFrame contains single ID column.

    Parameters
    ----------
        df : pd.DataFrame
            DataFrame to be validated.

    Raises
    -------
        ValueError
            If DataFrame contains multiple IDs.
    """
    raise_if(len(df["ID"].unique()) != 1, "DataFrame must have a single ID column.")


def _calculate_n_train(n_samples: int, test_size: Union[float, int]) -> int:
    """Calculate the number of train samples.

    Parameters
    ----------
        n_samples : int
            Number of samples in the DataFrame.
        test_size : float, int
            The percentage or number of samples to be used for validation.
            If the value is between 0 and 1, it is interpreted as a percentage of the total number of samples.
            If the value is greater than or equal to 1, it is interpreted as the number of samples to be used for validation.

    Returns
    -------
        int
            Number of train samples to be used in the split.

    Raises
    -------
        ValueError
            If test size is not a float in range (0.0, 1.0) or an integer < len(df).
    """
    if 0.0 < test_size < 1.0:
        n_valid = max(1, int(n_samples * test_size))
    else:
        raise_if(
            not (isinstance(test_size, int) and 1 < test_size < n_samples),
            "Test size should be a float in range (0.0, " "1.0) or an integer < len(df)",
        )
        n_valid = test_size

    return int(n_samples - n_valid)


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
        else:
            # Split data according to time threshold defined by the valid_p
            threshold_time_stamp = find_time_threshold(df, test_percentage)
            df_train, df_val = split_considering_timestamp(df, threshold_time_stamp)
    # df_train and df_val are returned as pd.DataFrames
    return df_train, df_val


def __crossvalidation_split_df(df, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            data with single ID columns
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
    total_samples = len(df)
    samples_per_fold, samples_overlap = _calculate_cv_params(total_samples, k, fold_pct, fold_overlap_pct)
    folds = []
    df_fold = df.copy(deep=True)
    for i in range(k, 0, -1):
        df_train, df_val = split_df(df_fold, test_percentage=samples_per_fold)
        folds.append((df_train, df_val))
        split_idx = len(df_fold) - samples_per_fold + samples_overlap
        df_fold = df_fold.iloc[:split_idx].reset_index(drop=True)
    folds = folds[::-1]
    return folds


def _calculate_cv_params(total_samples: int, k: int, fold_pct: float, fold_overlap_pct: float) -> Tuple[int, int]:
    """Return validated cross validation arguments.

    Parameters
    ----------
        total_samples : int
            number of data samples
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds

    Returns
    -------
        tuple (samples_per_fold, samples_overlap)

            samples fold

            samples overlap

    Raises
    -------
        ValueError
            If samples overlap is bigger than samples fold.
        ValueError
            If test percentage too large and there are not enough train samples.
    """
    samples_per_fold = max(1, int(fold_pct * total_samples))
    samples_overlap = int(fold_overlap_pct * samples_per_fold)
    raise_if(samples_overlap > samples_per_fold, "Samples overlap is bigger than samples fold")

    min_train = total_samples - samples_per_fold - (k - 1) * (samples_per_fold - samples_overlap)
    raise_if(
        min_train < samples_per_fold,
        "Test percentage too large. Not enough train samples. Select smaller test " "percentage.",
    )
    return samples_per_fold, samples_overlap


def _crossvalidation_split_df(
    df, received_single_time_series, k, fold_pct, fold_overlap_pct=0.0, global_model_cv_type="global-time"
):
    """Splits data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            data
        received_single_time_series : bool
            Whether the data contains a single time series
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
    Raises
    -------
        ValueError
            If invalid type of crossvalidation is selected.
    """
    if received_single_time_series:
        folds = (
            df.groupby("ID")
            .apply(lambda x: __crossvalidation_split_df(x, k=k, fold_pct=fold_pct, fold_overlap_pct=fold_overlap_pct))
            .tolist()[0]
        )

    else:
        # implement procedure for multiple IDs
        if global_model_cv_type == "global-time" or global_model_cv_type is None:
            # Use time threshold to perform crossvalidation (the distribution of data of different episodes may not be equivalent)
            folds = _crossvalidation_with_time_threshold(df, k=k, fold_pct=fold_pct, fold_overlap_pct=fold_overlap_pct)
        elif global_model_cv_type == "local":
            # Crossvalidate time series locally (time leakage may be a problem)
            folds_dict = (
                df.groupby("ID")
                .apply(
                    lambda x: __crossvalidation_split_df(x, k=k, fold_pct=fold_pct, fold_overlap_pct=fold_overlap_pct)
                )
                .to_dict()
            )
            folds = unfold_dict_of_folds(folds_dict, k)

        elif global_model_cv_type == "intersect":
            # Use data only from the time period of intersection among time series
            folds_dict = {}
            # Check for intersection of time so time leakage does not occur among different time series
            start_date, end_date = find_valid_time_interval_for_cv(df)
            for df_name, df_i in df.groupby("ID"):
                mask = (df_i["ds"] >= start_date) & (df_i["ds"] <= end_date)
                df_i = df_i[mask].copy(deep=True)
                folds_dict[df_name] = __crossvalidation_split_df(
                    df_i, k=k, fold_pct=fold_pct, fold_overlap_pct=fold_overlap_pct
                )
            folds = unfold_dict_of_folds(folds_dict, k)
        else:
            raise ValueError(
                "Please choose a valid type of global model crossvalidation (i.e. global-time, local, or intersect)"
            )
        pass

    return folds


def crossvalidation_split_df(
    df, received_single_time_series, global_model_cv_type, k=5, fold_pct=0.1, fold_overlap_pct=0.5
):
    """Splits timeseries data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
        received_ID_col : bool
            Whether the data contains an ID column
        received_single_time_series : bool
            Whether the data contains a single time series
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

    folds = _crossvalidation_split_df(
        df,
        k=k,
        fold_pct=fold_pct,
        fold_overlap_pct=fold_overlap_pct,
        received_single_time_series=received_single_time_series,
        global_model_cv_type=global_model_cv_type,
    )
    # ID col is kept for further processing
    return folds


def _crossvalidation_with_time_threshold(df, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation accordingly to time threshold.

    Parameters
    ----------
        df : pd.DataFrame
            data with column ``ds``, ``y``, and ``ID``
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
    df_merged = merge_dataframes(df)
    samples_per_fold, samples_overlap = _calculate_cv_params(len(df_merged), k, fold_pct, fold_overlap_pct)
    folds = []
    df_fold, _, _, _ = prep_or_copy_df(df)
    for i in range(k, 0, -1):
        threshold_time_stamp = find_time_threshold(df_fold, samples_per_fold)
        df_train, df_val = split_considering_timestamp(df_fold, threshold_time_stamp=threshold_time_stamp)
        folds.append((df_train, df_val))
        split_idx = len(df_merged) - samples_per_fold + samples_overlap
        df_merged = df_merged[:split_idx].reset_index(drop=True)
        threshold_time_stamp = df_merged["ds"].iloc[-1]
        df_fold_aux = pd.DataFrame()
        for df_name, df_i in df_fold.groupby("ID"):
            df_aux = (
                df_i.copy(deep=True).iloc[: len(df_i[df_i["ds"] < threshold_time_stamp]) + 1].reset_index(drop=True)
            )
            df_fold_aux = pd.concat((df_fold_aux, df_aux), ignore_index=True)
        df_fold = df_fold_aux.copy(deep=True)
    folds = folds[::-1]
    return folds


def _double_crossvalidation_split_df(df, k, valid_pct, test_pct):
    """Splits data in two sets of k folds for crossvalidation on validation and test data.

    Parameters
    ----------
        df : pd.DataFrame
            data
        k : int
            number of CV folds
        valid_pct : float
            percentage of overall samples to be in validation
        test_pct : float
            percentage of overall samples to be in test

    Returns
    -------
        tuple of k tuples [(folds_val, folds_test), …]
            elements same as :meth:`crossvalidation_split_df` returns
    """
    if len(df["ID"].unique()) > 1:
        raise NotImplementedError("double_crossvalidation_split_df not implemented for df with many time series")
    fold_pct_test = float(test_pct) / k
    folds_test = crossvalidation_split_df(df, k, fold_pct=fold_pct_test, fold_overlap_pct=0.0)
    df_train = folds_test[0][0]
    fold_pct_val = float(valid_pct) / k / (1.0 - test_pct)
    folds_val = crossvalidation_split_df(df_train, k, fold_pct=fold_pct_val, fold_overlap_pct=0.0)
    return folds_val, folds_test


def double_crossvalidation_split_df(self, df, k=5, valid_pct=0.10, test_pct=0.10):
    """Splits timeseries data in two sets of k folds for crossvalidation on training and testing data.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
        k : int
            number of CV folds
        valid_pct : float
            percentage of overall samples to be in validation
        test_pct : float
            percentage of overall samples to be in test

    Returns
    -------
        tuple of k tuples [(folds_val, folds_test), …]
            elements same as :meth:`crossvalidation_split_df` returns
    """
    folds_val, folds_test = _double_crossvalidation_split_df(
        df,
        k=k,
        valid_pct=valid_pct,
        test_pct=test_pct,
    )
    return folds_val, folds_test


def merge_dataframes(df: pd.DataFrame) -> pd.DataFrame:
    """Join dataframes for procedures such as splitting data, set auto seasonalities, and others.

    Parameters
    ----------
        df : pd.DataFrame
            containing column ``ds``, ``y``, and ``ID`` with data

    Returns
    -------
        pd.Dataframe
            Dataframe with concatenated time series (sorted 'ds', duplicates removed, index reset)

    Raises
    -------
        ValueError
            If df is not an instance of pd.DataFrame.
        ValueError
            If df does not contain 'ID' column.

    """
    raise_if(not isinstance(df, pd.DataFrame), "Can not join other than pd.DataFrames")
    raise_if("ID" not in df.columns, "df does not contain 'ID' column")

    df_merged = df.copy(deep=True).drop("ID", axis=1)
    df_merged = df_merged.sort_values("ds")
    df_merged = df_merged.drop_duplicates(subset=["ds"])
    df_merged = df_merged.reset_index(drop=True)
    return df_merged


def find_time_threshold(df, valid_p):
    """Find time threshold for dividing timeseries into train and validation sets.
    Prevents overbleed of targets. Overbleed of inputs can be configured.

    Parameters
    ----------
        df : pd.DataFrame
            data with column ``ds``, ``y``, and ``ID``
        valid_p : float
            fraction (0,1) of data to use for holdout validation set

    Returns
    -------
        str
            time stamp threshold defines the boundary for the train and validation sets split.
    """
    df_merged = merge_dataframes(df)
    n_samples = len(df_merged)
    n_train = _calculate_n_train(n_samples, valid_p)

    threshold_time_stamp = df_merged.loc[n_train, "ds"]
    log.debug("Time threshold: ", threshold_time_stamp)
    return threshold_time_stamp


def split_considering_timestamp(df, threshold_time_stamp):
    """Splits timeseries into train and validation sets according to given threshold_time_stamp.

    Parameters
    ----------
        df : pd.DataFrame
            data with column ``ds``, ``y``, and ``ID``
        threshold_time_stamp : str
            time stamp boundary that defines splitting of data

    Returns
    -------
        pd.DataFrame, dict
            training data
        pd.DataFrame, dict
            validation data
    """
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        if df[df["ID"] == df_name]["ds"].max() < threshold_time_stamp:
            df_train = pd.concat((df_train, df_i.copy(deep=True)), ignore_index=True)
        elif df[df["ID"] == df_name]["ds"].min() > threshold_time_stamp:
            df_val = pd.concat((df_val, df_i.copy(deep=True)), ignore_index=True)
        else:
            df_aux = df_i.copy(deep=True)
            n_train = len(df_aux[df_aux["ds"] < threshold_time_stamp])
            split_idx_train = n_train
            split_idx_val = split_idx_train
            df_train = pd.concat((df_train, df_aux.iloc[:split_idx_train]), ignore_index=True)
            df_val = pd.concat((df_val, df_aux.iloc[split_idx_val:]), ignore_index=True)
    return df_train, df_val


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
    ValueError
        If the dataframe does not have at least `min_len` rows.
    """
    raise_if(
        df.groupby("ID").apply(lambda x: len(x) < min_len).any(),
        "Input time series has not enough sample to " "fit an predict the model.",
    )


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
        pd.DataFrame,
            The modified 'predicted' dataframe.
    """
    predicted_new = predicted.groupby("ID").apply(lambda x: x[samples:]).reset_index(drop=True)
    return predicted_new


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
    _validate_single_ID_df(df)

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

    Raises
    -------
        ValueError
            If Dataframe has no rows.
        ValueError
            If Dataframe does not have columns 'ds' with the dates.
        ValueError
            If NaN is found in column 'ds'.
        ValueError
            If column 'ds' has timezone specified, which is not supported.
        ValueError
            If column 'ds' has duplicate values.
    """
    _validate_single_ID_df(df)

    raise_if(df.shape[0] == 0, "Dataframe has no rows.")
    raise_if("ds" not in df, 'Dataframe must have columns "ds" with the dates.')
    raise_if(df.loc[:, "ds"].isnull().any(), "Found NaN in column ds.")

    if df["ds"].dtype == np.int64:
        df["ds"] = df.loc[:, "ds"].astype(str)
    if pd.api.types.is_string_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df.loc[:, "ds"])
    if not np.issubdtype(df["ds"].dtype, np.datetime64):
        df["ds"] = pd.to_datetime(df.loc[:, "ds"])

    raise_if(df["ds"].dt.tz is not None, "Column ds has timezone specified, which is not supported. Remove timezone.")
    raise_if(len(df.ds.unique()) != len(df.ds), "Column ds has duplicate values. Please remove duplicates.")

    columns = []
    if check_y:
        columns.append("y")

    for name in columns:
        raise_if(name not in df, f"Column {name!r} missing from dataframe")
        raise_if(df.loc[df.loc[:, name].notnull()].shape[0] < 1, f"Dataframe column {name!r} only has NaN rows.")
        if not np.issubdtype(df[name].dtype, np.number):
            df.loc[:, name] = pd.to_numeric(df.loc[:, name])
        if np.isinf(df.loc[:, name].values).any():
            df.loc[:, name] = df[name].replace([np.inf, -np.inf], np.nan)
        raise_if(df.loc[df.loc[:, name].notnull()].shape[0] < 1, f"Dataframe column {name!r} only has NaN rows.")

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
        pd.DataFrame
            df with ID col
        bool
            whether the ID col was present
        bool
            whether it is a single time series

    Raises
    -------
        ValueError
            If df is None.
        ValueError
            If df type is invalid.
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
        _validate_single_ID_df(df)
        new_df.drop("ID", axis=1, inplace=True)
        log.info("Returning df with no ID column")
    return new_df


def unfold_dict_of_folds(folds_dict, k):
    """Convert dict of folds for typical format of folding of train and test data.

    Parameters
    ----------
        folds_dict : dict
            dict of folds
        k : int
            number of folds initially set

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
    Raises
    -------
        DataValidationError
            If number of folds in folds_dict does not correspond to k.
    """
    folds = []
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for j in range(0, k):
        for key in folds_dict:
            raise_data_validation_error_if(
                k != len(folds_dict[key]), "Number of folds in folds_dict does not " "correspond to k"
            )
            df_train = pd.concat((df_train, folds_dict[key][j][0]), ignore_index=True)
            df_test = pd.concat((df_test, folds_dict[key][j][1]), ignore_index=True)
        folds.append((df_train, df_test))
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
    return folds


def find_valid_time_interval_for_cv(df):
    """Find time interval of interception among all the time series from dict.

    Parameters
    ----------
        df : pd.DataFrame
            data with column ``ds``, ``y``, and ``ID``

    Returns
    -------
        str
            time interval start
        str
            time interval end
    """
    # Creates first time interval based on data from first key
    time_interval_intersection = df[df["ID"] == df["ID"].iloc[0]]["ds"]
    for df_name, df_i in df.groupby("ID"):
        time_interval_intersection = pd.merge(time_interval_intersection, df_i, how="inner", on=["ds"])
        time_interval_intersection = time_interval_intersection[["ds"]]
    start_date = time_interval_intersection["ds"].iloc[0]
    end_date = time_interval_intersection["ds"].iloc[-1]
    return start_date, end_date
