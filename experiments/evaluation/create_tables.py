import os
import pandas as pd
import pathlib
from openpyxl.styles import Alignment
import openpyxl
from openpyxl.utils import get_column_letter
import numpy as np
import pandas as pd
from experiments.evaluation.utils.params import (
    get_default_scaler,
    get_default_weighted,
    get_default_scaling_level,
    get_default_norm_type,
    get_default_norm_affine,
)

parent_dir = pathlib.Path(__file__).parent.absolute()
tables_dir = os.path.join(parent_dir, "tables")
results_merged = pd.read_csv(os.path.join(tables_dir, "results_merged.csv"), dtype={'norm_affine': str})
file_name_csv = os.path.join(tables_dir, "df_transposed_by_exp_id.csv")
file_name_xlsx = os.path.join(tables_dir, "all_tables.xlsx")
import numpy as np
import pandas as pd


# function that creates a DataFrame is transposed by exp_id and has all scaling and norm combinations as columns
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
    df, model_ids, metrics=["MASE", "MAE", "RMSE"], data_group_id=["SEA", "SEASH", "TRE", "STRU", "HET"], scaler= get_default_scaler(), weighted= get_default_weighted(), scaling_level= get_default_scaling_level(), norm_type= get_default_norm_type(), norm_affine= get_default_norm_affine()
):
    # Create an IndexSlice object to slice the multi-index
    idx = pd.IndexSlice

    # Filter the DataFrame based on the specified conditions
    filtered_df = df.loc[
        idx[model_ids, data_group_id, :],
        idx[metrics, scaler, weighted, scaling_level, norm_type, norm_affine],
    ]
    return filtered_df


def find_best_scaler_methods(df, models, metric):
    # Filter the DataFrame based on the specified models and scaling levels
    filtered_cols = [
        col
        for col in df.columns[2:]
        if col.split("/")[3] in ["per_time_series", "per_dataset"]
        and col.split("/")[2] == "None"
        and col.split("/")[0] == metric
    ]
    filtered_cols = np.append(filtered_cols, df.columns[:2])
    filtered_df = df.loc[:, filtered_cols]

    # Get the unique combinations of exp_id and model_id
    model_related_exps = df[df["model_id"] == models]

    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=["exp_id", "model_id", "scaler_method", "improvement"])

    # Iterate over the unique combinations of exp_id and model_id
    for _, row in model_related_exps.iterrows():
        exp_id = row["exp_id"]
        model_id = models

        combination_df = filtered_df[(filtered_df["exp_id"] == exp_id) & (filtered_df["model_id"] == model_id)]
        df_all = df[(df["exp_id"] == exp_id) & (df["model_id"] == model_id)]

        # Check if all values in the combination DataFrame are NaN
        if combination_df.iloc[:, :-2].isna().all().all():
            best_scaler_method = np.nan
            improvement = np.nan
        else:
            # Find the best scaler method based on the specified metric
            best_scaler_method = combination_df.iloc[:, :-2].idxmin(axis=1, skipna=True)
            best_scaler = best_scaler_method.str.split("/").str[1]
            best_scaling_level = best_scaler_method.str.split("/").str[3]
            best_scaling_level_pendant = (
                "per_time_series" if best_scaling_level.values[0] == "per_dataset" else "per_dataset"
            )

            # Find the improvement relative to the 'no scaler_None_None_None_False' entry
            no_scaler_metric = df_all[metric + "/no scaler/None/None/None/False"].values[0]
            improvement = no_scaler_metric / combination_df.loc[:, best_scaler_method].values[0]

            # Calculate the improvement for the pendant scaling level
            pendant_method = combination_df.columns[
                combination_df.columns.str.contains(best_scaler.values[0])
                & combination_df.columns.str.contains(best_scaling_level_pendant)
            ]
            pendant_improvement = no_scaler_metric / combination_df.loc[:, pendant_method].values[0]

        # Append the result to the output DataFrame
        result_df = result_df.append(
            {
                "exp_id": exp_id,
                "model_id": model_id,
                "scaler_method": best_scaler_method,
                "improvement": improvement,
                "pendant_improvement": pendant_improvement,
            },
            ignore_index=True,
        )

    return result_df


###main ###
df_transposed_by_exp_id, df_transpby_exp_id_flat = create_transposed_df_by_model_id_and_exp_id(results_merged)
# df_best_scaler = find_best_scaler_methods(df_transposed_by_exp_id, 'NP_FNN', 'MAE')
df_filtered = filter_df_by_model_id_metric_and_data_group_id_and_rows(
    df_transposed_by_exp_id,
    model_ids=["LGBM"],
    metrics=["MASE", "MAE", "RMSE"],
    data_group_id = ['TRE'],
    scaler = ['LogTransformer()'],
)
# save results
df_transposed_by_exp_id.to_csv(file_name_csv, index=False)
with pd.ExcelWriter(file_name_xlsx, engine="xlsxwriter") as writer:
    sheet_names = ["df_transpby_exp_id_flat"]
    for sheet_name in sheet_names:
        df_transpby_exp_id_flat.to_excel(writer, sheet_name=sheet_name)
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        # Add a header format.
        worksheet.freeze_panes(1, 0)
        # # Set autofilter
        worksheet.autofilter(0, 1, results_merged.shape[0], results_merged.shape[1])

#
#         # save results
#         with pd.ExcelWriter('tables/model_specific_metrics.xlsx', engine='xlsxwriter') as writer:
#             for i, df in enumerate(results):
#                 df.to_excel(writer, sheet_name=id_models[i])
#
#                 # Get workbook and worksheet objects
#                 workbook = writer.book
#                 worksheet = writer.sheets[id_models[i]]
#
#                 # Define your format objects
#                 format_bad = workbook.add_format({'bg_color': '#FF0000'})  # Red
#                 format_neutral = workbook.add_format({'bg_color': '#FFA500'})  # Orange
#                 format_false = workbook.add_format({'bg_color': '#008000'})  # Green
#                 format_true = workbook.add_format({'bg_color': '#FF0000'})  # Red
#
#                 # Apply conditional formatting
#                 # Adjust column references as needed
#                 worksheet.conditional_format('K2:K1000', {'type': 'text',
#                                                           'criteria': 'containing',
#                                                           'value': 'BAD',
#                                                           'format': format_bad})
#                 worksheet.conditional_format('K2:K1000', {'type': 'text',
#                                                           'criteria': 'containing',
#                                                           'value': 'NEUTRAL',
#                                                           'format': format_neutral})
#                 worksheet.conditional_format('K2:K1000', {'type': 'text',
#                                                           'criteria': 'containing',
#                                                           'value': 'FALSE',
#                                                           'format': format_false})
#                 worksheet.conditional_format('I2:I1000', {'type': 'text',
#                                                           'criteria': 'containing',
#                                                           'value': 'TRUE',
#                                                           'format': format_true})
#                 # Freeze the first row
#                 worksheet.freeze_panes(1, 0)
#
#                 # Set autofilter
#                 worksheet.autofilter(0, 1, df.shape[0], df.shape[1])
#
#     return results


BASELINE_EPERIMENTS = [
    "gen_one_shape_ar_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_None_None",
    "gen_cancel_shape_ar_n_ts_[5, 5]_am_[10, 10]_of_[0, 0]_gr_None_None",
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]",  # has different name and 2 names
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]_None",
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]",
    "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]",  # has 2 names
    "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None",
    "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]_None",  # has 2 names
    "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]",
]
BASELINE_EPERIMENTS_WINDOW_BASED = [
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]",  # has different name and 2 names
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]_None",
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]",
    "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]",  # has 2 names
    "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None",
    "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]_None",  # has 2 names
    "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]",
]
ALL_MODELS = ["NP_localST_", "NP_FNN_", "NP_", "TP_localST_", "TP_", "LGBM_", "RNN_", "TF_", "SNaive_", "Naive_"]
ALL_MODELS_WINDOW_BASED = ["NP_FNN_sw_wb_", "NP_FNN_wb_", "RNN_wb_nl_", "RNN_wb_"]

# avg_overview_df = create_grouped_df(
#     selected_rows,
#     selected_id_exp=BASELINE_EPERIMENTS,
#     selected_id_model=ALL_MODELS,
#     metric = 'MASE',
#     window_based = False,
# )
#
# # Save to file
# avg_overview_df.to_csv(os.path.join(parent_dir, "avg_overview_df.csv"), index=False)
# avg_overview_df.to_excel(os.path.join(parent_dir, "avg_overview_df.xlsx"))
#
# avg_overview_window_based_df = create_grouped_df(
#     selected_rows,
#     selected_id_exp=BASELINE_EPERIMENTS_WINDOW_BASED,
#     selected_id_model=ALL_MODELS_WINDOW_BASED,
#     metric = 'MASE',
#     window_based = True,
# )
# # Save to file
# avg_overview_window_based_df.to_csv(os.path.join(parent_dir, "avg_overview_window_based_df.csv"), index=False)
# avg_overview_window_based_df.to_excel(os.path.join(parent_dir, "avg_overview_window_based_df.xlsx"))


# model_specific_metrics = filter_selected_rows(selected_rows, id_models=ALL_MODELS, group_list=['SEA', 'SEASH', 'TRE', 'STRU', 'HET'], metric='MASE')
# Save to file
# model_specific_metrics.to_csv(os.path.join(parent_dir, "model_specific_metrics.csv"), index=False)
# model_specific_metrics.to_excel(os.path.join(parent_dir, "model_specific_metrics.xlsx"))
