import os
import pandas as pd
from shutil import copyfile
import pathlib

parent_dir = pathlib.Path(__file__).parent.absolute()
res_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "results")
save_path = os.path.join(parent_dir, "tables")
file_name_csv = os.path.join(save_path, "results_merged_raw.csv")
file_name_xlsx = os.path.join(save_path, "results_merged_raw.xlsx")

# Make sure the directories exist
os.makedirs(save_path, exist_ok=True)


dfs_list = []

for data_group in os.listdir(res_path):
    data_group_path = os.path.join(res_path, data_group)
    for exp in os.listdir(data_group_path):
        exp_path = os.path.join(data_group_path, exp)
        if os.path.isdir(data_group_path):
            results_csv_path = os.path.join(exp_path, "results.csv")
            if os.path.isfile(results_csv_path):
                # copy file to separate folder
                # target_exp_path = os.path.join(save_path, f"{exp}_results.csv")
                # copyfile(results_csv_path, target_exp_path)

                # read csv
                df = pd.read_csv(results_csv_path)

                # round the number to 4 decimal places
                df = df.round(4)

                # add ID column
                df['exp_id'] = exp
                df['data_group_id'] = data_group
                df['scaling_level'] = df['scaling level']
                df = df.drop('scaling level', axis=1)
                df = df.drop('data', axis=1)

                # append to the list
                dfs_list.append(df)

# concatenate all dataframes
df_merged = pd.concat(dfs_list, ignore_index=True).reset_index(drop=True)
df_merged = df_merged.drop('Unnamed: 0', axis=1)
# replace nan in col norm_affine and norm_type with None
df_merged['norm_affine'] = df_merged['norm_affine'].fillna('False')
df_merged['norm_affine'] = df_merged['norm_affine'].replace('none', 'False')
df_merged['norm_affine'] = df_merged['norm_affine'].replace(False, 'False')
df_merged['norm_affine'] = df_merged['norm_affine'].replace(True, 'True')
df_merged['norm_type'] = df_merged['norm_type'].fillna('None')
df_merged['norm_type'] = df_merged['norm_type'].replace('none', 'None')
df_merged['weighted'] = df_merged['weighted'].fillna('None')
df_merged['weighted'] = df_merged['weighted'].replace('none', 'None')
df_merged['scaling_level'] = df_merged['scaling_level'].fillna('None')
df_merged['scaling_level'] = df_merged['scaling_level'].replace('none', 'None')
df_merged['scaler'] = df_merged['scaler'].fillna('None')
df_merged['scaler'] = df_merged['scaler'].replace('none', 'None')
df_merged['scaler'] = df_merged['scaler'].replace('no scaler', 'None')

# write to a csv file
df_merged.to_excel(file_name_xlsx, index=False)
df_merged.to_csv(file_name_csv, index=False)




