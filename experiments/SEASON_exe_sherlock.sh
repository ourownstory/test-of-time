#!/bin/bash

### NP
commands_NP=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### NP_localST
commands_NP_localST=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### NP_FNN
commands_NP_FNN=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### TP
commands_TP=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### TP_localST
comannds_TP_localST=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### TF
commands_TF=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### RNN
commands_RNN=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### LGBM
commands_LGBM=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### Naive
commands_Naive=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

### SNaive
commands_SNaive=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5"'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5"'
)

# combine the command lists
python_commands=("${commands_NP[@]}" "${commands_NP_localST[@]}" "${commands_NP_FNN[@]}" "${commands_TP[@]}" "${commands_TP_localST[@]}" "${commands_TF[@]}" "${commands_RNN[@]}" "${commands_LGBM[@]}" "${commands_Naive[@]}" "${commands_SNaive[@]}")

# initialize job counter
job_counter=1

# loop through the python commands
for command in "${python_commands[@]}"; do
    # create a job name based on the counter
    job_name="job_$job_counter"

    # create a temporary Slurm script
    echo "#!/bin/bash" > temp.sh
    echo "#SBATCH --job-name=$job_name" >> temp.sh
    echo "#SBATCH --output=res_$job_name" >> temp.sh
    echo "#SBATCH --time=00:40:00" >> temp.sh
    echo "#SBATCH --cpus-per-task=19" >> temp.sh
    echo "#SBATCH --mem=2G" >> temp.sh

    # add the python command to the Slurm script
    echo $command >> temp.sh

    # submit the Slurm job
    sbatch temp.sh

    # remove the temporary Slurm script
    rm temp.sh

    # increment the job counter
    ((job_counter++))
done


