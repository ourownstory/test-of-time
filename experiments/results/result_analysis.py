import os
import pandas as pd
from shutil import copyfile
import pathlib

parent_dir = pathlib.Path(__file__).parent.absolute()
df_results_all = pd.read_csv(os.path.join(parent_dir, "results_all.csv"))
# Exclude 'QuantileTransformer' rows when 'ID_GROUP' is 'TRE'
df_results_all = df_results_all[~((df_results_all['ID_GROUP'] == 'TRE') & (df_results_all['scaler'] == 'QuantileTransformer'))]
df_results_all_reset = df_results_all.reset_index(drop=True)
# Drop 'Unnamed' and 'data' columns (if necessary)
df_results_all_reset = df_results_all.drop(columns=['Unnamed: 0', 'data'], errors='ignore')

# Replace model names with empty string
model_names = [ "NP_localST_", "NP_FNN_sw_wb_", "NP_FNN_wb_", "NP_FNN_", "NP_", "TP_localST_", "TP_", "LGBM_", "RNN_wb_nl_", "RNN_wb_", "RNN_", "TF_",  "SNaive_", "Naive_"]


# Copy the DataFrame to avoid altering the original DataFrame
df_results_all_reset_modified = df_results_all_reset.copy()

for model in model_names:
    mask = df_results_all_reset_modified['ID_EXP'].str.contains(model)
    df_results_all_reset_modified.loc[mask, 'ID_EXP'] = df_results_all_reset_modified.loc[mask, 'ID_EXP'].str.replace(model, '', regex=False)
    df_results_all_reset_modified.loc[mask, 'ID_model'] = model

df_results_all = df_results_all_reset_modified.copy()

# Define a mapping of keywords to new group values
keyword_to_group = {
    "_trend_": "TRE",
    "hetero": "HET",
    "struc_break": "STRU"
}

# Iterate over the keywords and their corresponding new group values
for keyword, new_group in keyword_to_group.items():
    # Create a boolean mask indicating rows where 'ID_GROUP' is 'WIN' and 'ID_EXP' contains the keyword
    mask = (df_results_all['ID_GROUP'] == 'WIN') & (df_results_all['ID_EXP'].str.contains(keyword))

    # Use the mask to replace 'WIN' with the new group value in the 'ID_GROUP' column
    df_results_all.loc[mask, 'ID_GROUP'] = new_group

# Temporary: Set the 'norm_mode' column to 'instance' where 'ID_model' equals 'RNN_wb'
df_results_all.loc[df_results_all['ID_model'] == 'RNN_wb_', 'norm_mode'] = 'instance'

# Temporary: Set the 'norm_mode' column to 'batch' where 'ID_model' equals 'RNN_wb_nl'
df_results_all.loc[df_results_all['ID_model'] == 'RNN_wb_nl_', 'norm_mode'] = 'batch'


### ANALYSIS ###
# Define a function to select the best and 2nd best scaler
def select_best_scalers(df):
    df = df.sort_values(by='MASE', ascending=True)
    # if df only has one row, return the first row
    if len(df) == 1:
        return df.iloc[[0]]
    # if df has at least two rows, return the first two rows
    else:
        return df.iloc[[0, 1]]
# Create a function to calculate percentage improvement
def calculate_percentage_improvement(df, metric):
    no_scaler_metric = no_scaler_rows.loc[(no_scaler_rows['ID_model'] == df['ID_model']) & (no_scaler_rows['ID_EXP'] == df['ID_EXP']), metric].values[0]
    return ((no_scaler_metric - df[metric]) / no_scaler_metric) * 100

# Extract 'no scaler' rows
no_scaler_rows = df_results_all[df_results_all['scaler'] == 'no scaler'].copy()
no_scaler_rows['rank'] = 'no scaler'

# Apply function to each group for per_time_series scaling level
best_scalers_rows_series = df_results_all[df_results_all['scaling_level'] == 'per_time_series'].groupby(['ID_model', 'ID_EXP']).apply(select_best_scalers).reset_index(drop=True).copy()

# Reset the index
best_scalers_rows_series = best_scalers_rows_series.reset_index(drop=True).drop(columns=['Unnamed: 0', 'data'], errors='ignore')

# Apply function to each group for per_dataset scaling level
best_scalers_rows_dataset = df_results_all[df_results_all['scaling_level'] == 'per_dataset'].groupby(['ID_model', 'ID_EXP']).apply(select_best_scalers).reset_index(drop=True).copy()
# best_scalers_rows_dataset['rank'] = 'best per_time_series'

# Reset the index
best_scalers_rows_dataset = best_scalers_rows_dataset.reset_index(drop=True).drop(columns=['Unnamed: 0', 'data'], errors='ignore')

# Select the best and 2nd best scalers for per_time_series scaling level
best_scaler_rows_series = best_scalers_rows_series.groupby(['ID_model', 'ID_EXP']).first().reset_index()
best_scaler_rows_series['rank'] = 'best per_time_series'
second_best_scaler_rows_series = best_scalers_rows_series.groupby(['ID_model', 'ID_EXP']).nth(1).reset_index()
second_best_scaler_rows_series['rank'] = '2nd best per_time_series'


# Select the best and 2nd best scalers for per_dataset scaling level
best_scaler_rows_dataset = best_scalers_rows_dataset.groupby(['ID_model', 'ID_EXP']).first().reset_index()
best_scaler_rows_dataset['rank'] = 'best per_dataset'
second_best_scaler_rows_dataset = best_scalers_rows_dataset.groupby(['ID_model', 'ID_EXP']).nth(1).reset_index()
second_best_scaler_rows_dataset['rank'] = '2nd best per_dataset'


# Add column to indicate whether per-series is not effective
best_scaler_rows_series['per_series_not_effective'] = best_scaler_rows_dataset['MASE'] <= (best_scaler_rows_series['MASE'] + 0.05)
second_best_scaler_rows_series['per_series_not_effective'] = best_scaler_rows_dataset['MASE'] <= (second_best_scaler_rows_series['MASE'] + 0.05)


# Calculate MASE percentage improvement
best_scaler_rows_series['MASE_%_improvement'] = best_scaler_rows_series.apply(calculate_percentage_improvement, args=('MASE',), axis=1)
second_best_scaler_rows_series['MASE_%_improvement'] = second_best_scaler_rows_series.apply(calculate_percentage_improvement, args=('MASE',), axis=1)

# Calculate MAE percentage improvement
best_scaler_rows_series['MAE_%_improvement'] = best_scaler_rows_series.apply(calculate_percentage_improvement, args=('MAE',), axis=1)
second_best_scaler_rows_series['MAE_%_improvement'] = second_best_scaler_rows_series.apply(calculate_percentage_improvement, args=('MAE',), axis=1)

# Add column to check if MASE improved more than MAE
best_scaler_rows_series['MASE_improved_more'] = abs(best_scaler_rows_series['MASE_%_improvement'] / best_scaler_rows_series['MAE_%_improvement']) > 2
second_best_scaler_rows_series['MASE_improved_more'] = abs(second_best_scaler_rows_series['MASE_%_improvement'] / second_best_scaler_rows_series['MAE_%_improvement']) > 2

# Concatenate all the rows
selected_rows = pd.concat([no_scaler_rows, best_scaler_rows_series, second_best_scaler_rows_series, best_scaler_rows_dataset, second_best_scaler_rows_dataset ], axis=0).reset_index(drop=True)

# Add 'per-series not effective' to the remaining rows
# selected_rows['per_series_not_effective'] = selected_rows['per_series_not_effective'].fillna(False)

# Identify the best model for each experiment
best_model_per_experiment = selected_rows.groupby('ID_EXP')['MASE'].idxmin()


# Extract the model and MASE for each experiment's best model
best_model_and_MASE_per_experiment = selected_rows.loc[best_model_per_experiment, ['ID_model', 'MASE', 'ID_EXP']]

# Create a new DataFrame that maps each experiment ID to the model and MASE of the best model
experiment_winner_df = best_model_and_MASE_per_experiment.reset_index().rename(columns={'MASE': 'experiment_winner'})
experiment_winner_df = experiment_winner_df.drop(columns=['index', 'ID_model'], errors='ignore')

# Merge the new DataFrame with the original DataFrame
selected_rows = pd.merge(selected_rows, experiment_winner_df, on='ID_EXP', how='left')



# Save to file
selected_rows.to_csv(os.path.join(parent_dir, "selected_rows.csv"), index=False)
selected_rows.to_excel(os.path.join(parent_dir, "selected_rows.xlsx"), index=False)



