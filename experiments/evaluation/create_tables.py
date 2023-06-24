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
    get_baseline_experiments,
    get_model_params_list,
    get_grouped_experiments,
get_model_groups,
)
import pandas as pd
from experiments.evaluation.utils.helper import flatten_dict_values, add_subgroup_col, average_over_exp_subgroups, pivot_exp_subgroup_metrics, pivot_best_scaler_view\
    , calculate_best_scaler, create_transposed_df_by_model_id_and_exp_id, filter_df_by_model_id_metric_and_data_group_id_and_rows, average_over_models, calculate_improvement


parent_dir = pathlib.Path(__file__).parent.absolute()
tables_dir = os.path.join(parent_dir, "tables")
results_merged = pd.read_csv(os.path.join(tables_dir, "results_merged.csv"), dtype={'norm_affine': str})
file_name_csv = os.path.join(tables_dir, "df_transposed_by_exp_id.csv")
file_name_xlsx = os.path.join(tables_dir, "all_tables.xlsx")


def view_best_scaler_type_and_value_per_data_group_and_model(results_merged):
    df_transpby_exp_id,_ = create_transposed_df_by_model_id_and_exp_id(results_merged)
    df_filtered = filter_df_by_model_id_metric_and_data_group_id_and_rows(df_transpby_exp_id, exps_ids=get_baseline_experiments())
    df_best_scaler = calculate_best_scaler(df_filtered, metrics=["MASE"], scope='all')
    df_best_scaler_view = pivot_best_scaler_view(df_best_scaler)
    return df_best_scaler_view

def view_per_grouped_exp_and_model(results_merged, scaler_scope='all', metrics=["MASE", "MAE", "RMSE"]):
    '''
    scaler_scope: 'all', 'best', 'grouped_models'
    '''
    df_transposed_by_exp_id, _ = create_transposed_df_by_model_id_and_exp_id(results_merged)
    grouped_experiment_dict = get_grouped_experiments()
    grouped_models_dict = get_model_groups()
    grouped_experiments = flatten_dict_values(grouped_experiment_dict)
    df_filtered = filter_df_by_model_id_metric_and_data_group_id_and_rows(
        df_transposed_by_exp_id,
        exp_ids=grouped_experiments,
        metrics=metrics,
        )
    if scaler_scope == 'best':
        df_filtered_improvements = calculate_best_scaler(df_filtered, metrics=metrics, scope='all')
    else:
        df_filtered_improvements = calculate_improvement(df_filtered, metrics=metrics, scope='all')
    df_with_subgroups = add_subgroup_col(df_filtered_improvements, grouped_experiment_dict, grouped_models_dict)
    df_averaged_per_subgroup = average_over_exp_subgroups(df_with_subgroups, scaler_scope=scaler_scope, metrics=metrics)
    if scaler_scope == 'grouped_models':
        df_averaged_per_subgroup = average_over_models(df_averaged_per_subgroup, metrics=metrics, model_scope='all')
    df_pivoted = pivot_exp_subgroup_metrics(df_averaged_per_subgroup, scaler_scope, metrics)
    return df_pivoted
def view_best_baseline(results_merged, metrics=["MASE", "MAE", "RMSE"]):
    df_transposed, _ = create_transposed_df_by_model_id_and_exp_id(results_merged)
    df_filtered = filter_df_by_model_id_metric_and_data_group_id_and_rows(df_transposed, exp_ids=get_baseline_experiments(), metrics=metrics)
    df_best = calculate_best_scaler(df_filtered, metrics=metrics, scope='best')
    df_pivoted = pivot_best_scaler_view(df_best)
    return df_pivoted

###main ###
per_model_group_all_normalized_MASE = view_per_grouped_exp_and_model(results_merged, scaler_scope='grouped_models', metrics=["MASE"])
per_model_best_MASE = view_per_grouped_exp_and_model(results_merged, scaler_scope='best', metrics=["MASE"])
per_model_group_and_data_group_best_MASE = view_best_baseline(results_merged, metrics=['MASE'])
df_list = [per_model_group_all_normalized_MASE, per_model_best_MASE, per_model_group_and_data_group_best_MASE ]

### save results ###
with pd.ExcelWriter(file_name_xlsx, engine="xlsxwriter") as writer:
    sheet_names = [str(i) for i in range(len(df_list))]
    for i, sheet_name in enumerate(sheet_names):
        df_list[i].to_excel(writer, sheet_name=sheet_name)
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        # Add a header format.
        # worksheet.freeze_panes(1, 0)
        # # Set autofilter
        # worksheet.autofilter(0, 1, results_merged.shape[0], results_merged.shape[1])
#





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



