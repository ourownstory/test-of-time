import os
import pandas as pd
from shutil import copyfile
import pathlib

parent_dir = pathlib.Path(__file__).parent.absolute()
res_path = os.path.join(parent_dir, "../../../results/res")
extract_path = parent_dir
# merged_path = os.path.join(parent_dir, "results_all.csv")

# Make sure the directories exist
os.makedirs(extract_path, exist_ok=True)
# os.makedirs(merged_path, exist_ok=True)

all_data = []

for dir_name in os.listdir(res_path):
    dir_path = os.path.join(res_path, dir_name)
    for res_dir_name in os.listdir(dir_path):
        res_dir_path = os.path.join(dir_path, res_dir_name)
        if os.path.isdir(dir_path):
            results_csv_path = os.path.join(res_dir_path, "results.csv")
            if os.path.isfile(results_csv_path):
                # copy file to separate folder
                extract_file_path = os.path.join(extract_path, f"{res_dir_name}_results.csv")
                copyfile(results_csv_path, extract_file_path)

                # read csv
                df = pd.read_csv(results_csv_path)

                # round the number to 4 decimal places
                df = df.round(4)

                # add ID column
                df['ID_EXP'] = res_dir_name
                df['ID_GROUP'] = dir_name
                df['scaling_level'] = df['scaling level']
                df = df.drop('scaling level', axis=1)

                # append to the list
                all_data.append(df)

# concatenate all dataframes
merged_df = pd.concat(all_data, ignore_index=True)

# write to a csv file
merged_csv_path = os.path.join(parent_dir, "results_all.csv")
merged_df.to_excel(merged_csv_path.replace(".csv", ".xlsx"), index=False)
merged_df.to_csv(merged_csv_path, index=False)




