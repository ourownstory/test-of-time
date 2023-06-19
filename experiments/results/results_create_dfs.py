import os
import pandas as pd
import pathlib

parent_dir = pathlib.Path(__file__).parent.absolute()
selected_rows = pd.read_csv(os.path.join(parent_dir, "selected_rows.csv"))


def create_grouped_df(selected_rows, selected_id_exp=None, selected_id_model=None, metric = 'MASE', window_based=False):
    # Set default values
    if selected_id_exp is None:
        selected_id_exp = selected_rows['ID_EXP'].unique().tolist()
    if selected_id_model is None:
        selected_id_model = selected_rows['ID_model'].unique().tolist()

    # Filter selected_rows based on the selected ID_EXP and ID_model
    selected_rows_filtered = selected_rows[
        selected_rows['ID_EXP'].isin(selected_id_exp) & selected_rows['ID_model'].isin(selected_id_model)]
    # Define the groups
    if window_based:
        groups = ['TRE', 'STRU', 'HET']
        scaler_types_list = ['no scaler', 'instance', 'batch']
        selection_column = 'norm_mode'
    else:
        groups = ['SEA', 'SEASH', 'TRE', 'STRU', 'HET']
        scaler_types_list = ['no scaler', 'best per_time_series', 'best per_dataset']
        selection_column = 'rank'

    # Initialize an empty DataFrame to store the result
    result_df = pd.DataFrame()

    for group in groups:
        for scaler_type in scaler_types_list:
            # Filter the rows corresponding to the current group and scaler type
            filtered_rows = selected_rows_filtered[
                (selected_rows_filtered['ID_GROUP'] == group) & (selected_rows_filtered[selection_column] == scaler_type)]

            # Calculate the average MASE for each model
            avg_metric = filtered_rows.groupby('ID_model')[metric].mean()

            # Add the average MASE to the result DataFrame
            result_df = pd.concat([result_df, avg_metric], axis=1)

    # Rename the columns
    result_df.columns = pd.MultiIndex.from_product([groups, scaler_types_list])

    return result_df

def check_problem(row):
    if row['best per_time_series_MASE'] < row['no scaler_MASE'] - 0.08:
        return 'WAHR'
    elif row['no scaler_MASE'] <= row['experiment_winner'] + 0.1:
        return 'FALSE'
    elif row['best per_time_series_MASE'] < row['no scaler_MASE'] and row['best per_time_series_MASE'] >= row['no scaler_MASE'] - 0.05:
        return 'NEUTRAL'
    else:
        return 'BAD'

def filter_selected_rows(selected_rows, id_models, group_list, metric):
    # Initialize a list to store the results for each model
    results = []

    # Loop through each model in the list
    for id_model in id_models:
        # Filter rows by the current model
        rows_filtered_by_model = selected_rows[selected_rows['ID_model'] == id_model]

        # Initialize an empty DataFrame to store the results for the current model
        result_df = pd.DataFrame()

        # Loop through each group in the list
        for group in group_list:
            # Filter rows by the current group
            rows_filtered_by_group = rows_filtered_by_model[rows_filtered_by_model['ID_GROUP'] == group]

            # Initialize a DataFrame to store the metrics for the current group
            group_df = pd.DataFrame()

            # Extract the necessary metrics for each scaler type and add them to group_df
            for scaler_type in ['no scaler', 'best per_time_series', 'best per_dataset']:
                # Filter rows by the current scaler type
                rows_filtered_by_scaler = rows_filtered_by_group[rows_filtered_by_group['rank'] == scaler_type]

                # Extract the metric and rename the column
                metric_series = rows_filtered_by_scaler[metric].rename(f"{scaler_type}_{metric}").reset_index(drop=True)

                # If group_df is empty, add the metric as a new column; otherwise, merge with the existing DataFrame
                if group_df.empty:
                    group_df = pd.concat([rows_filtered_by_scaler[['ID_EXP', 'ID_model','ID_GROUP', 'experiment_winner']].reset_index(drop=True), metric_series], axis=1).reset_index(drop=True)
                else:
                    group_df = pd.concat([group_df, metric_series], axis=1)

            # Extract the 'per_series_not_effective' value for the best per_series scaler
            per_series_not_effective = rows_filtered_by_group[rows_filtered_by_group['rank'] == 'best per_time_series']['per_series_not_effective']
            per_series_scaler_type = rows_filtered_by_group[rows_filtered_by_group['rank'] == 'best per_time_series']['scaler']

            group_df['per_series_not_effective'] = per_series_not_effective.values
            group_df['per_series_scaler_type'] = per_series_scaler_type.values
            group_df['has_problem'] = group_df.apply(check_problem, axis=1)

            # Append the group_df DataFrame to the result DataFrame
            result_df = result_df.append(group_df)

        # Reset the index of the result DataFrame
        result_df = result_df.reset_index(drop=True)

        # Append the result DataFrame to the list of results
        results.append(result_df)
        # save results
        with pd.ExcelWriter('model_specific_metrics.xlsx', engine='xlsxwriter') as writer:
            for i, df in enumerate(results):
                df.to_excel(writer, sheet_name=id_models[i])

                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets[id_models[i]]

                # Define your format objects
                format_bad = workbook.add_format({'bg_color': '#FF0000'})  # Red
                format_neutral = workbook.add_format({'bg_color': '#FFA500'})  # Orange
                format_false = workbook.add_format({'bg_color': '#008000'})  # Green
                format_true = workbook.add_format({'bg_color': '#FF0000'})  # Red

                # Apply conditional formatting
                # Adjust column references as needed
                worksheet.conditional_format('K2:K1000', {'type': 'text',
                                                          'criteria': 'containing',
                                                          'value': 'BAD',
                                                          'format': format_bad})
                worksheet.conditional_format('K2:K1000', {'type': 'text',
                                                          'criteria': 'containing',
                                                          'value': 'NEUTRAL',
                                                          'format': format_neutral})
                worksheet.conditional_format('K2:K1000', {'type': 'text',
                                                          'criteria': 'containing',
                                                          'value': 'FALSE',
                                                          'format': format_false})
                worksheet.conditional_format('I2:I1000', {'type': 'text',
                                                          'criteria': 'containing',
                                                          'value': 'TRUE',
                                                          'format': format_true})

    return results




BASELINE_EPERIMENTS = [
        'gen_one_shape_ar_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_None_None',
        'gen_cancel_shape_ar_n_ts_[5, 5]_am_[10, 10]_of_[0, 0]_gr_None_None',
        'gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]', # has different name and 2 names
        'gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]_None',
        'gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]',
        'gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]', # has 2 names
        'gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None',
        'gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]_None',  # has 2 names
        'gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]',
]
BASELINE_EPERIMENTS_WINDOW_BASED = [
        'gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]', # has different name and 2 names
        'gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]_None',
        'gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]',
        'gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]', # has 2 names
        'gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None',
        'gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]_None',  # has 2 names
        'gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]',
]
ALL_MODELS = [ "NP_localST_", "NP_FNN_", "NP_", "TP_localST_", "TP_", "LGBM_",  "RNN_", "TF_",  "SNaive_", "Naive_"]
ALL_MODELS_WINDOW_BASED = [ "NP_FNN_sw_wb_", "NP_FNN_wb_", "RNN_wb_nl_", "RNN_wb_"]

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


model_specific_metrics = filter_selected_rows(selected_rows, id_models=ALL_MODELS, group_list=['SEA', 'SEASH', 'TRE', 'STRU', 'HET'], metric='MASE')
# Save to file
# model_specific_metrics.to_csv(os.path.join(parent_dir, "model_specific_metrics.csv"), index=False)
# model_specific_metrics.to_excel(os.path.join(parent_dir, "model_specific_metrics.xlsx"))
