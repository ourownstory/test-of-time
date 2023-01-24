import logging

import numpy as np
import pandas as pd
from neuralprophet.df_utils import prep_or_copy_df

log = logging.getLogger("tot.df_utils")


def reshape_raw_predictions_to_forecast_df(df, predicted, n_req_past_observations, n_req_future_observations):
    """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

    Parameters
    ----------
        df : pd.DataFrame
            input dataframe
        predicted : np.array
            Array containing the predictions

    Returns
    -------
        pd.DataFrame
            columns ``ds``, ``y``, optionally ``ID`` and [``yhat<i>``],

            Note
            ----
            where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
            e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
    """
    cols = ["ds", "y", "ID"]  # cols to keep from df
    fcst_df = pd.concat((df[cols],), axis=1)
    # create a line for each forecast_lag
    # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
    for forecast_lag in range(1, n_req_future_observations + 1):
        forecast = predicted[:, forecast_lag - 1]
        pad_before = n_req_past_observations + forecast_lag - 1
        pad_after = n_req_future_observations - forecast_lag
        yhat = np.concatenate(
            ([np.NaN] * pad_before, forecast, [np.NaN] * pad_after)
        )  # add pad based on n_forecasts and current forecast_lag
        name = f"yhat{forecast_lag}"
        fcst_df[name] = yhat

    return fcst_df


def _split_df(df, test_percentage):
    """Splits timeseries df into train and validation sets.
    Parameters
    ----------
        df : pd.DataFrame
            data to be splitted
        test_percentage : float, int
    Returns
    -------
        pd.DataFrame
            training data
        pd.DataFrame
            validation data
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


def split_df(df, test_percentage=0.25, local_split=False):
    """Splits timeseries df into train and validation sets.
    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        test_percentage : float, int
            fraction (0,1) of data to use for holdout validation set, or number of validation samples >1
    Returns
    -------
        pd.DataFrame
            training data
        pd.DataFrame
            validation data
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

    # df_train and df_val are returned as pd.DataFrames
    return df_train, df_val


def _crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0):
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
    assert min_train >= samples_fold
    folds = []
    df_fold = df.copy(deep=True)
    for i in range(k, 0, -1):
        df_train, df_val = split_df(df_fold, test_percentage=samples_fold)
        folds.append((df_train, df_val))
        split_idx = len(df_fold) - samples_fold + samples_overlap
        df_fold = df_fold.iloc[:split_idx].reset_index(drop=True)
    folds = folds[::-1]
    return folds


def df_util_crossvalidation_split_df(
    df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0, global_model_cv_type="global-time"
):
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
    df, _, _, _ = prep_or_copy_df(df)
    if len(df["ID"].unique()) == 1:
        for df_name, df_i in df.groupby("ID"):
            folds = _crossvalidation_split_df(df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
    # else:
    #     if global_model_cv_type == "global-time" or global_model_cv_type is None:
    #         # Use time threshold to perform crossvalidation (the distribution of data of different episodes may not be equivalent)
    #         folds = _crossvalidation_with_time_threshold(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
    #     elif global_model_cv_type == "local":
    #         # Crossvalidate time series locally (time leakage may be a problem)
    #         folds_dict = {}
    #         for df_name, df_i in df.groupby("ID"):
    #             folds_dict[df_name] = _crossvalidation_split_df(
    #                 df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct
    #             )
    #         folds = unfold_dict_of_folds(folds_dict, k)
    #     elif global_model_cv_type == "intersect":
    #         # Use data only from the time period of intersection among time series
    #         folds_dict = {}
    #         # Check for intersection of time so time leakage does not occur among different time series
    #         start_date, end_date = find_valid_time_interval_for_cv(df)
    #         for df_name, df_i in df.groupby("ID"):
    #             mask = (df_i["ds"] >= start_date) & (df_i["ds"] <= end_date)
    #             df_i = df_i[mask].copy(deep=True)
    #             folds_dict[df_name] = _crossvalidation_split_df(
    #                 df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct
    #             )
    #         folds = unfold_dict_of_folds(folds_dict, k)
    #     else:
    #         raise ValueError(
    #             "Please choose a valid type of global model crossvalidation (i.e. global-time, local, or intersect)"
    #         )
    return folds


def crossvalidation_split_df(
    n_lags, n_forecasts, df, freq, k=5, fold_pct=0.1, fold_overlap_pct=0.5, global_model_cv_type="global-time"
):
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
        global_model_cv_type : str
            Type of crossvalidation to apply to the dict of time series.

                options:

                    ``global-time`` (default) crossvalidation is performed according to a timestamp threshold.

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
        double_crossvalidation_split_df : Splits timeseries data in two sets of k folds for crossvalidation on training and testing data.

    Examples
    --------
        >>> df1 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-01', periods = 10, freq = 'D'),
        ...                     'y': [9.59, 8.52, 8.18, 8.07, 7.89, 8.09, 7.84, 7.65, 8.71, 8.09]})
        >>> df2 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-02', periods = 10, freq = 'D'),
        ...                     'y': [8.71, 8.09, 7.84, 7.65, 8.02, 8.52, 8.18, 8.07, 8.25, 8.30]})
        >>> df3 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-03', periods = 10, freq = 'D'),
        ...                     'y': [7.67, 7.64, 7.55, 8.25, 8.32, 9.59, 8.52, 7.55, 8.25, 8.09]})
        >>> df3
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

    You can create folds for a single dataframe.
        >>> folds = m.crossvalidation_split_df(df3, k = 2, fold_pct = 0.2)
        >>> folds
        [(  ds            y
            0 2022-12-03  7.67
            1 2022-12-04  7.64
            2 2022-12-05  7.55
            3 2022-12-06  8.25
            4 2022-12-07  8.32
            5 2022-12-08  9.59
            6 2022-12-09  8.52,
            ds            y
            0 2022-12-10  7.55
            1 2022-12-11  8.25),
        (   ds            y
            0 2022-12-03  7.67
            1 2022-12-04  7.64
            2 2022-12-05  7.55
            3 2022-12-06  8.25
            4 2022-12-07  8.32
            5 2022-12-08  9.59
            6 2022-12-09  8.52
            7 2022-12-10  7.55,
            ds            y
            0 2022-12-11  8.25
            1 2022-12-12  8.09)]

    We can also create a df with many IDs.
        >>> df1['ID'] = 'data1'
        >>> df2['ID'] = 'data2'
        >>> df3['ID'] = 'data3'
        >>> df = pd.concat((df1, df2, df3))

    When using the df with many IDs, there are three types of possible crossvalidation. The default crossvalidation is performed according to a timestamp threshold. In this case, we can have a different number of samples for each time series per fold. This approach prevents time leakage.
        >>> folds = m.crossvalidation_split_df(df, k = 2, fold_pct = 0.2)
    One can notice how each of the folds has a different number of samples for the validation set. Nonetheless, time leakage does not occur.
        >>> folds[0][1]
            ds	y	ID
        0	2022-12-10	8.09	data1
        1	2022-12-10	8.25	data2
        2	2022-12-11	8.30	data2
        3	2022-12-10	7.55	data3
        4	2022-12-11	8.25	data3
        >>> folds[1][1]
            ds	y	ID
        0	2022-12-11	8.30	data2
        1	2022-12-11	8.25	data3
        2	2022-12-12	8.09	data3
    In some applications, crossvalidating each of the time series locally may be more adequate.
        >>> folds = m.crossvalidation_split_df(df, k = 2, fold_pct = 0.2, global_model_cv_type = 'local')
    In this way, we prevent a different number of validation samples in each fold.
        >>> folds[0][1]
            ds	y	ID
        0	2022-12-08	7.65	data1
        1	2022-12-09	8.71	data1
        2	2022-12-09	8.07	data2
        3	2022-12-10	8.25	data2
        4	2022-12-10	7.55	data3
        5	2022-12-11	8.25	data3
        >>> folds[1][1]
            ds	y	ID
        0	2022-12-09	8.71	data1
        1	2022-12-10	8.09	data1
        2	2022-12-10	8.25	data2
        3	2022-12-11	8.30	data2
        4	2022-12-11	8.25	data3
        5	2022-12-12	8.09	data3
    The last type of global model crossvalidation gets the time intersection among all the time series used. There is no time leakage in this case, and we preserve the same number of samples per fold. The only drawback of this approach is that some of the samples may not be used (those not in the time intersection).
        >>> folds = m.crossvalidation_split_df(df, k = 2, fold_pct = 0.2, global_model_cv_type = 'intersect')
        >>> folds[0][1]
            ds	y	ID
        0	2022-12-09	8.71	data1
        1	2022-12-09	8.07	data2
        2	2022-12-09	8.52	data3
        0 2022-12-09  8.52}
        >>> folds[1][1]
            ds	y	ID
        0	2022-12-10	8.09	data1
        1	2022-12-10	8.25	data2
        2	2022-12-10	7.55	data3
    """
    df, received_ID_col, received_single_time_series, _ = prep_or_copy_df(df)
    # df = self._check_dataframe(df, check_y=False, exogenous=False) #add later
    # freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq) #add later
    # df = model._handle_missing_data(df, freq=freq, predicting=False) #should be implemented, pass the model
    folds = df_util_crossvalidation_split_df(
        df,
        n_lags=n_lags,  # TODO: needs to generalize
        n_forecasts=n_forecasts,
        k=k,
        fold_pct=fold_pct,
        fold_overlap_pct=fold_overlap_pct,
        global_model_cv_type=global_model_cv_type,
    )
    if not received_ID_col and received_single_time_series:
        # Delete ID column (__df__) of df_train and df_val of all folds in case ID was not previously provided
        for i in range(len(folds)):
            del folds[i][0]["ID"]
            del folds[i][1]["ID"]
    return folds
