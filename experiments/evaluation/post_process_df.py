import os
import pandas as pd
import pathlib
from experiments.evaluation.utils.params import get_model_params_list,get_all_model_params_list, get_data_group_keyword, get_rnn_norm_type_keyword, get_window_based_model_name_keyword
import numpy as np


parent_dir = pathlib.Path(__file__).parent.absolute()
tables_dir = os.path.join(parent_dir, "tables")
results_merged = pd.read_csv(os.path.join(tables_dir, "results_merged_raw.csv"), dtype={'norm_affine': str})
file_name_csv = os.path.join(tables_dir, "results_merged.csv")
file_name_xlsx = os.path.join(tables_dir, "results_merged.xlsx")


def remove_selected_scaler_from_df(df, scaler):
    df_filtered = df[df['scaler'] != scaler]
    return df_filtered
def rename_minmax_scaler(df):
    df['scaler'] = df['scaler'].replace('MinMaxScaler(feature_range=(-0.5, 0.5))', 'MinMaxScaler()')
    return df

def pop_model_name_from_exp_name(df, model_names):
    df = df.assign(model_id='')
    for model in model_names:
        mask = df['exp_id'].str.contains(model)
        df.loc[mask, 'exp_id'] = df.loc[mask, 'exp_id'].str.replace(model+'_', '', regex=False)
        df.loc[mask, 'model_id'] = model
    return df

def remove_pytorch_norm_mode(df):
    df_filtered = df[df['norm_mode'] != 'pytorch']
    df_filtered = df_filtered.drop('norm_mode', axis=1)
    return df_filtered

def replace_window_based_data_group(df, data_group_keywords):
    for keyword, new_group in data_group_keywords.items():
        mask = (df['data_group_id'] == 'WIN') & (df['exp_id'].str.contains(keyword))
        df.loc[mask, 'data_group_id'] = new_group
    return df

def replace_window_based_model_names(df, window_based_model_name_keywords):
    for keyword, new_name in window_based_model_name_keywords.items():
        mask = df['model_id'].str.contains(keyword)
        df.loc[mask, 'model_id'] = new_name
    return df

def set_norm_type_for_RNN(df, RNN_norm_type_keywords):
    for RNN_norm_type_keyword, norm_mode in RNN_norm_type_keywords.items():
        mask = df['model_id'].str.contains(RNN_norm_type_keyword)
        df.loc[mask, 'norm_type'] = norm_mode
        df.loc[mask, 'norm_affine'] = False
    return df
def remove_last_underscore_in_experimet(df):
    df['exp_id'] = df['exp_id'].str.rstrip('_')
    return df
def remove_last_None_in_experimet(df):
    mask = df['data_group_id'].str.contains('TRE') | df['data_group_id'].str.contains('HET') | df['data_group_id'].str.contains('WIN') | df['data_group_id'].str.contains('SEA') | df['data_group_id'].str.contains('SEASH')

    df['exp_id'] = df['exp_id'].str.rstrip('_None')
    return df

def drop_duplicates_minmax(df):
    # drop duplicate rows for scaler == MinMaxScaler, either keep rows with MAE col non-nan or the first row
    df = df.sort_values(by=['exp_id', 'data_group_id', 'model_id', 'scaling_level', 'weighted', 'scaler'], ascending=[True, True, True, True, True, True])
    df = df.drop_duplicates(subset=['exp_id', 'data_group_id', 'model_id', 'scaling_level', 'weighted', 'scaler', 'model', 'norm_type', 'norm_affine'], keep='first')
    return df



# main
results_merged = results_merged.copy()
results_merged = remove_selected_scaler_from_df(results_merged, "QuantileTransformer(output_distribution='normal')")
results_merged = rename_minmax_scaler(results_merged)
results_merged = pop_model_name_from_exp_name(results_merged, model_names=get_all_model_params_list())
results_merged = replace_window_based_data_group(results_merged, data_group_keywords = get_data_group_keyword())
results_merged = remove_pytorch_norm_mode(results_merged)
results_merged = set_norm_type_for_RNN(results_merged, RNN_norm_type_keywords=get_rnn_norm_type_keyword())
results_merged = replace_window_based_model_names(results_merged, window_based_model_name_keywords=get_window_based_model_name_keyword())
results_merged = remove_last_underscore_in_experimet(results_merged)
results_merged = remove_last_None_in_experimet(results_merged)
results_merged = drop_duplicates_minmax(results_merged)

# save results
results_merged.to_csv(file_name_csv, index=False)
with pd.ExcelWriter(file_name_xlsx, engine='xlsxwriter') as writer:
    results_merged.to_excel(writer, sheet_name='results_merged')
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['results_merged']
    # Add a header format.
    worksheet.freeze_panes(1, 0)
    # Set autofilter
    worksheet.autofilter(0, 1, results_merged.shape[0], results_merged.shape[1])


