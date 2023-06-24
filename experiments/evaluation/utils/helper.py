import numpy as np
import pandas as pd
from experiments.evaluation.utils.params import (
    get_default_scaler,
    get_default_weighted,
    get_default_scaling_level,
    get_default_norm_type,
    get_default_norm_affine,
    get_baseline_experiments,
    get_model_params_list,
    get_grouped_experiments,
)

def create_transposed_df_by_model_id_and_exp_id(df_stacked):
    # Create a new column 'scaler_methods' by combining the values from multiple columns
    df_stacked["scaler_methods"] = (
        df_stacked["scaler"].astype(str)
        + "/"
        + df_stacked["weighted"].astype(str)
        + "/"
        + df_stacked["scaling_level"].astype(str)
        + "/"
        + df_stacked["norm_type"].astype(str)
        + "/"
        + df_stacked["norm_affine"].astype(str)
    )

    # Group df_stacked by 'exp_id' and 'model_id', and aggregate MAE and MASE values for each 'scaler_methods'
    df_pivoted = (
        df_stacked.groupby(
            [
                "exp_id",
                "data_group_id",
                "model_id",
                "scaler",
                "weighted",
                "scaling_level",
                "norm_type",
                "norm_affine",
                "scaler_methods",
            ]
        )
        .agg({"MAE": "first", "MASE": "first", "RMSE": "first"})
        .reset_index()
    )

    # Create a new DataFrame by pivoting 'scaler_methods' to column names
    df_transposed = df_pivoted.pivot(
        index=["model_id", "data_group_id", "exp_id"],
        columns=["scaler", "weighted", "scaling_level", "norm_type", "norm_affine"],
        values=["MAE", "MASE", "RMSE"],
    )
    # reset index
    # Flatten the column names by combining 'MAE' and 'scaler_method', and 'MASE' and 'scaler_method'
    column_names = [
        f"{metric}/{scaler}/{weighted}/{scaling_level}/{norm_type}/{norm_affine}"
        for metric, scaler, weighted, scaling_level, norm_type, norm_affine in df_transposed.columns
    ]
    df_transposed_flattened = df_transposed.copy()
    df_transposed_flattened.columns = column_names
    df_transposed_flattened = df_transposed_flattened.reset_index()

    return df_transposed, df_transposed_flattened


def filter_df_by_model_id_metric_and_data_group_id_and_rows(
    df, model_ids=get_model_params_list(), exp_ids = "all", metrics=["MASE", "MAE", "RMSE"], data_group_id=["SEA", "SEASH", "TRE", "STRU", "HET"], scaler= get_default_scaler(), weighted= get_default_weighted(), scaling_level= get_default_scaling_level(), norm_type= get_default_norm_type(), norm_affine= get_default_norm_affine()
):
    # Create an IndexSlice object to slice the multi-index
    idx = pd.IndexSlice

    # Filter the DataFrame based on the specified conditions
    filtered_df = df.loc[
        idx[model_ids, data_group_id, :] if exp_ids == "all" else idx[model_ids, data_group_id, exp_ids],
        idx[metrics, scaler, weighted, scaling_level, norm_type, norm_affine],
    ]
    return filtered_df

def calculate_best_scaler(df, metrics=["MASE", "MAE", "RMSE"], scope='all'):
    # Create an IndexSlice object to slice the multi-index
    idx = pd.IndexSlice
    norm_type = ["instance", "batch"]
    norm_affine = 'True'
    scaling_level = ["per_dataset", "per_time_series"]

    if scope == 'window_based':
        filtered_df = df.loc[:, idx[metrics, :, 'None', :, norm_type, norm_affine]]
    elif scope == 'non_window_based':
        filtered_df = df.loc[:, idx[metrics, :, 'None', scaling_level, :, :]]
    else:
        filtered_df = df.loc[:, idx[metrics, :, 'None', :, :, :]]
        filtered_df = filtered_df.drop(columns=filtered_df.loc[:, idx[metrics, "None", "None", "None", "None", "False"]])

    result_dfs = pd.DataFrame()
    for metric in metrics:
        filtered_df_per_metric = filtered_df.loc[:, idx[metric, :, :, :, :, :]]

        # Find the best non-window-based scalers
        best_scalers = filtered_df_per_metric.apply(lambda x: x.idxmin(), axis=1)
        best_values = filtered_df_per_metric.apply(lambda x: x.min(), axis=1)

        # Filter the DataFrame for no-scaler values and reset the index
        no_scaler_values = df.loc[:, idx[metric, "None", "None", "None", "None", "False"]]

        # Calculate no scaler-related improvement row-wise
        improvement = pd.Series( no_scaler_values.values.squeeze() / best_values.values.squeeze(), index=no_scaler_values.index)

        # Copy best_non_window_based_scalers and replace the 4th position with the other scaling_level
        pendant_scalers = best_scalers.copy()
        pendant_scalers = pendant_scalers.apply(
            lambda x: (x[0], x[1], x[2], x[3], 'batch' if x[4] == 'instance' else 'instance', x[5])
            if x[4] in norm_type
            else (x[0], x[1], x[2], 'per_dataset' if x[3] == 'per_time_series' else 'per_time_series', x[4], x[5])
        )
        # Convert the Series to a DataFrame
        pendant_scalers = pd.DataFrame(pendant_scalers, columns=["pendant_scalers"])

        pendant_values = pd.Series(dtype=float, index=pendant_scalers.index)
        # Iterate over the rows and extract the pendant values
        for i, row in pendant_scalers.iterrows():
            pendant_values.at[i] = filtered_df.loc[i, row[0]]
            # pendant_non_window_based_scalers.at[i, 'pendant_values'] = pendant_values

        # Calculate no scaler-related improvement row-wise
        pendant_improvement = pd.Series(no_scaler_values.values.squeeze() / pendant_values.values.squeeze(),
                                index=no_scaler_values.index)

        # Initialize an empty DataFrame to store the results
        result_df = pd.DataFrame(columns=["best_scaler", "improvement", "pendant_improvement"])
        result_df["best_scaler"] = best_scalers
        result_df["improvement"] = improvement
        result_df["pendant_improvement"] = pendant_improvement

        metric_columns = pd.MultiIndex.from_product([[metric], result_df.columns])
        result_df.columns = metric_columns

        result_dfs = pd.concat([result_dfs, result_df], axis=1)
    return result_dfs


def convert_to_scaled_metric(value_col, no_scaler_values):
    improvement = no_scaler_values / value_col.values
    return improvement

def calculate_improvement(df, metrics=["MASE", "MAE", "RMSE"], scope='all'):
    # Create an IndexSlice object to slice the multi-index
    idx = pd.IndexSlice
    norm_type = ["instance", "batch"]
    norm_affine = 'True'
    scaling_level = ["per_dataset", "per_time_series"]

    if scope == 'window_based':
        filtered_df = df.loc[:, idx[metrics, :, 'None', :, norm_type, norm_affine]]
    elif scope == 'non_window_based':
        filtered_df = df.loc[:, idx[metrics, :, 'None', scaling_level, :, :]]
    else:
        filtered_df = df.loc[:, idx[metrics, :, 'None', :, :, :]]
        filtered_df = filtered_df.drop(columns=filtered_df.loc[:, idx[metrics, "None", "None", "None", "None", "False"]])
    filtered_improvements=pd.DataFrame()
    for metric in metrics:
        filtered_df_per_metric = filtered_df.loc[:, idx[metric, :, :, :, :, :]]
        no_scaler_values = df.loc[:, idx[metric, "None", "None", "None", "None", "False"]]
        # Calculate no scaler-related improvement row-wise
        filtered_improvements_per_metric=filtered_df_per_metric.apply(lambda x: convert_to_scaled_metric(x, no_scaler_values), axis=0)
        filtered_improvements = pd.concat([filtered_improvements, filtered_improvements_per_metric], axis=1)


        # Calculate no scaler-related improvement row-wise
        # improvement = pd.Series( no_scaler_values.values.squeeze() / best_values.values.squeeze(), index=no_scaler_values.index)


    return filtered_improvements_per_metric


def flatten_dict_values(d):
    values = []
    for value in d.values():
        if isinstance(value, dict):
            for inner_values in value.values():
                values.append(inner_values)
    values_flat = list(set().union(*values))
    return values_flat

def add_subgroup_col(df, exp_dict, model_dict):
    # Create a new column to store the subgroup information
    df['sub_group_id'] = np.nan
    df['sub_model_group_id'] = np.nan
    df.reset_index(inplace=True)

    # Iterate over the subgroups in GROUPED_EXPERIMENTS
    for subgroup, experiments in exp_dict.items():
        for exp_key, experiment_ids in experiments.items():
            df.loc[df['exp_id'].isin(experiment_ids),'sub_group_id'] = exp_key
    for model_group, models in model_dict.items():
            df.loc[df['model_id'].isin(models),'sub_model_group_id'] = model_group
    return df

def average_over_exp_subgroups(df, scaler_scope, metrics):
    idx = pd.IndexSlice
    if scaler_scope == 'all':
        df = df.groupby(['model_id', 'data_group_id', 'sub_group_id']).apply(lambda x: x.loc[:, idx[metrics, :, :, :, :]].mean())
    if scaler_scope ==  'best':
        df = df.groupby(['model_id', 'data_group_id', 'sub_group_id']).apply(lambda x: x.loc[:, idx[:, 'improvement']].mean())
    if scaler_scope == 'grouped_models':
        df = df.groupby(['sub_model_group_id', 'data_group_id', 'sub_group_id']).apply(lambda x: x.loc[:, idx[metrics, :, :, :, :]].mean())
    return df

def average_over_models(df,metrics, model_scope='all'):
    idx = pd.IndexSlice
    if model_scope == 'model_groups':
        df = df.groupby(['data_group_id','sub_group_id', 'sub_model_group_id']).apply(lambda x: x.loc[:, idx[metrics, :, :, :, :]].mean())
    if model_scope == 'all':
        df = df.groupby(['data_group_id','sub_group_id']).apply(lambda x: x.loc[:, idx[metrics, :, :, :, :]].mean())
        # add empt ycol sub_model_group_id
        df['sub_model_group_id'] = 'ALL'
    return df

def pivot_best_scaler_view(df):
    # Group df_stacked by 'exp_id' and 'model_id', and aggregate MAE and MASE values for each 'scaler_methods'
    idx = pd.IndexSlice
    df = df.reset_index()
    df_pivoted = (
        df.groupby(["model_id", "data_group_id"])
        .apply(lambda x: x.loc[:, idx[:, ["improvement", "best_scaler", "pendant_improvement"]]].reset_index(drop=True)).droplevel(level=2, axis=0).reset_index()
    )

    # Create a new DataFrame by pivoting 'scaler_methods' to column names
    df_transposed = df_pivoted.pivot(
        index=["model_id"],
        columns=["data_group_id"],
    )
    # sort the rows
    # df_transposed = df_transposed.sort_index(axis=0, level=0)
    # sort the columns by the data group id
    df_transposed = df_transposed.sort_index(axis=1, level=2)
    return df_transposed

def pivot_exp_subgroup_metrics(df, scaler_scope, metrics):
    # Group df_stacked by 'exp_id' and 'model_id', and aggregate MAE and MASE values for each 'scaler_methods'
    idx = pd.IndexSlice
    df = df.reset_index()
    if scaler_scope == 'all' :
        df.reset_index()
        # Create a new DataFrame by pivoting 'scaler_methods' to column names
        df_transposed = df.pivot(
            index=["data_group_id", "sub_group_id"],
            columns=["model_id"],
        )
        df_transposed = df_transposed.swaplevel(0, 4, axis=1)
        order_cols = ['None', 'StandardScaler()', 'RobustScaler(quantile_range=(5, 95))', 'MinMaxScaler()', 'PowerTransformer()', 'LogTransformer()']
        order_rows = ['SEA', 'SEASH', 'TRE', 'HET', 'STRU']
        df_transposed = df_transposed.reindex(columns=order_cols, level=1)
        df_transposed = df_transposed.reindex(index=order_rows, level=0)
    if scaler_scope == 'grouped_models':
        df_transposed =df
        # df = df.drop('model_id', axis=1)
        # Create a new DataFrame by pivoting 'scaler_methods' to column names
        # df_transposed = df.pivot(
        #     index=["data_group_id", "sub_group_id"],
        #     columns=["sub_model_group_id"],
        # )
        # df_transposed = df_transposed.swaplevel(0, 6, axis=1)
        order_cols_2 = ['StandardScaler()', 'RobustScaler(quantile_range=(5, 95))', 'MinMaxScaler()', 'PowerTransformer()', 'LogTransformer()', 'None']
        order_cols_1 = ['data_group_id', 'sub_group_id']
        # order_rows = ['SEA', 'SEASH', 'TRE', 'HET', 'STRU']
        df_transposed_1= df_transposed.reindex(columns=order_cols_1, level=0)
        df_transposed_2 = df_transposed.reindex(columns=order_cols_2, level=1)
        df_transposed = pd.concat([df_transposed_1, df_transposed_2], axis=1)
        # df_transposed = df_transposed.sort_values(by=df_transposed.loc[:, idx['data_group_id', :, :, :, :, :]], key=lambda x: x.map(lambda v: order_rows.index(v)))

    if scaler_scope == 'best':
        df.reset_index()
        # Create a new DataFrame by pivoting 'scaler_methods' to column names
        df_transposed = df.pivot(
            index=["data_group_id", "sub_group_id"],
            columns=["model_id"],
        )
        order = ['TP', 'TP_localST', 'NP', 'NP_localST', 'NP_FNN', 'NP_FNN_sw', 'RNN', 'LGBM', 'TF']
        df_transposed = df_transposed.reindex(columns=order, level=2)


    # sort the rows
    # df_transposed = df_transposed.sort_ index(axis=0, level=0)
    # sort the columns by the data group id
    # df_transposed = df_transposed.sort_index(axis=1, level=1)
    return df_transposed
