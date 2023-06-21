#!/bin/bash

# load modules
module load python/3.9.0
module load cuda/11.7.1

# activate the virtual environment
source ../../tot4/bin/activate

# re-install the packages for full length scalers
pip uninstall -y neuralprophet
pip install git+https://github.com/ourownstory/neural_prophet.git@normalization-layer
pip uninstall -y darts
pip install -git+https://github.com/LeonieFreisinger/darts.git@lgbm_for_server#egg=darts


### NP
commands_NP=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NeuralProphetModel --params NP --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset London --model NeuralProphetModel --params NP --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NeuralProphetModel --params NP --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NeuralProphetModel --params NP --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NeuralProphetModel --params NP --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model NeuralProphetModel --params NP --gen_func "gen_model_and_params_scalers_reweighting"'
)

### NP_localST
commands_NP_localST=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NeuralProphetModel --params NP_localST --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset London --model NeuralProphetModel --params NP_localST --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NeuralProphetModel --params NP_localST --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NeuralProphetModel --params NP_localST --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NeuralProphetModel --params NP_localST --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --data_path "../datasets/ETTh_panel.csv" --model NeuralProphetModel --params NP_localST --gen_func "gen_model_and_params_scalers_reweighting"'

)

### NP_FNN
commands_NP_FNN=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NeuralProphetModel --params NP_FNN --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset London --model NeuralProphetModel --params NP_FNN --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NeuralProphetModel --params NP_FNN --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NeuralProphetModel --params NP_FNN --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NeuralProphetModel --params NP_FNN --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model NeuralProphetModel --params NP_FNN --gen_func "gen_model_and_params_scalers_reweighting"'
)

### NP_FNN_sw
commands_NP_FNN_sw=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NeuralProphetModel --params NP_FNN_sw --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset London --model NeuralProphetModel --params NP_FNN_sw --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NeuralProphetModel --params NP_FNN_sw --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NeuralProphetModel --params NP_FNN_sw --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NeuralProphetModel --params NP_FNN_sw --gen_func "gen_model_and_params_scalers_reweighting"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model NeuralProphetModel --params NP_FNN_sw --gen_func "gen_model_and_params_scalers_reweighting"'
)


### TP
commands_TP=(
'python3 EXP_REAL_DATA.py --dataset EIA --model TorchProphetModel --params TP --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model TorchProphetModel --params TP --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model TorchProphetModel --params TP --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model TorchProphetModel --params TP --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model TorchProphetModel --params TP --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model TorchProphetModel --params TP --gen_func "gen_model_and_params_scalers"'
)

### TP_localST
commands_TP_localST=(
'python3 EXP_REAL_DATA.py --dataset EIA --model TorchProphetModel --params TP_localST --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model TorchProphetModel --params TP_localST --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model TorchProphetModel --params TP_localST --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model TorchProphetModel --params TP_localST --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model TorchProphetModel --params TP_localST --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model TorchProphetModel --params TP_localST --gen_func "gen_model_and_params_scalers"'
)

### TF
commands_TF=(
'python3 EXP_REAL_DATA.py --dataset EIA --model TransformerModel --params TF --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model TransformerModel --params TF --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model TransformerModel --params TF --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model TransformerModel --params TF --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model TransformerModel --params TF --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model TransformerModel --params TF --gen_func "gen_model_and_params_scalers"'

)

### RNN
commands_RNN=(
'python3 EXP_REAL_DATA.py --dataset EIA --model RNNModel --params RNN --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model RNNModel --params RNN --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model RNNModel --params RNN --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model RNNModel --params RNN --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model RNNModel --params RNN --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model RNNModel --params RNN --gen_func "gen_model_and_params_scalers"'


)

### LGBM
commands_LGBM=(
'python3 EXP_REAL_DATA.py --dataset EIA --model LightGBMModel --params LGBM --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model LightGBMModel --params LGBM --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model LightGBMModel --params LGBM --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model LightGBMModel --params LGBM --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model LightGBMModel --params LGBM --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model LightGBMModel --params LGBM --gen_func "gen_model_and_params_scalers"'

)

### Naive
commands_Naive=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NaiveModel --params Naive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model NaiveModel --params Naive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NaiveModel --params Naive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NaiveModel --params Naive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NaiveModel --params Naive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model NaiveModel --params Naive --gen_func "gen_model_and_params_scalers"'
)

### SNaive
commands_SNaive=(
'python3 EXP_REAL_DATA.py --dataset EIA --model SeasonalNaiveModel --params SNaive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset London --model SeasonalNaiveModel --params SNaive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model SeasonalNaiveModel --params SNaive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Australian --model SeasonalNaiveModel --params SNaive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset Solar --model SeasonalNaiveModel --params SNaive --gen_func "gen_model_and_params_scalers"'
'python3 EXP_REAL_DATA.py --dataset ETTH_panel --model SeasonalNaiveModel --params SNaive --gen_func "gen_model_and_params_scalers"'
)

# combine the command lists
python_commands=("${commands_NP[@]}" "${commands_NP_localST[@]}" "${commands_NP_FNN[@]}" "${commands_NP_FNN_sw[@]}" "${commands_TP[@]}" "${commands_TP_localST[@]}" "${commands_TF[@]}" "${commands_RNN[@]}" "${commands_LGBM[@]}" "${commands_Naive[@]}" "${commands_SNaive[@]}")

# initialize job counter
job_counter=1

# loop through the python commands
for command in "${python_commands[@]}"; do
    # create a job name based on the counter
    job_name="job_hsc_$job_counter"
    echo "Submitting $job_name"

    # create a temporary Slurm script
    echo "#!/bin/bash" > temp.sh
    echo "#SBATCH --job-name=$job_name" >> temp.sh
    echo "#SBATCH --output=res_$job_name" >> temp.sh

    # check if "Transformer" or "RNN" is in the command
    if [[ $command == *"Transformer"* ]] || [[ $command == *"RNN"* ]]; then
        echo "#SBATCH --time=06:00:00" >> temp.sh
        echo "#SBATCH -p gpu" >> temp.sh
        echo "#SBATCH -G 1" >> temp.sh
    else
        echo "#SBATCH --time=3:00:00" >> temp.sh
        echo "#SBATCH --cpus-per-task=10" >> temp.sh
        echo "#SBATCH --mem-per-cpu=1G" >> temp.sh
    fi

    # add the python command to the Slurm script
    echo $command >> temp.sh

    # submit the Slurm job
    sbatch temp.sh

    # remove the temporary Slurm script
    rm temp.sh

    # increment the job counter
    ((job_counter++))
done

pip uninstall -y darts
pip install git+https://github.com/LeonieFreisinger/darts.git@revin_nonlearnable#egg=darts

### NP_FNN_wb
commands_NP_FNN_wb=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NeuralProphetModel --params NP_FNN_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset London --model NeuralProphetModel --params NP_FNN_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NeuralProphetModel --params NP_FNN_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NeuralProphetModel --params NP_FNN_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NeuralProphetModel --params NP_FNN_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset custom --data_path "../datasets/ETTh_panel.csv" --model NeuralProphetModel --params NP_FNN_wb --gen_func "gen_model_and_params_norm"'
)

### NP_FNN_sw_wb
commands_NP_FNN_sw_wb=(
'python3 EXP_REAL_DATA.py --dataset EIA --model NeuralProphetModel --params NP_FNN_sw_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset London --model NeuralProphetModel --params NP_FNN_sw_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model NeuralProphetModel --params NP_FNN_sw_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset Australian --model NeuralProphetModel --params NP_FNN_sw_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset Solar --model NeuralProphetModel --params NP_FNN_sw_wb --gen_func "gen_model_and_params_norm"'
'python3 EXP_REAL_DATA.py --dataset custom --data_path "../datasets/ETTh_panel.csv" --model NeuralProphetModel --params NP_FNN_sw_wb --gen_func "gen_model_and_params_norm"'
)

### RNN_wb_in
commands_RNN_wb_in=(
'python3 EXP_REAL_DATA.py --dataset EIA --model RNNModel --params RNN_wb_in --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset London --model RNNModel --params RNN_wb_in --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model RNNModel --params RNN_wb_in --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset Australian --model RNNModel --params RNN_wb_in --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset Solar --model RNNModel --params RNN_wb_in --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset custom --data_path "../datasets/ETTh_panel.csv" --model RNNModel --params RNN_wb_in --gen_func "gen_model_and_params_none"'
)

# combine the command lists
python_commands_wn_1=( "${commands_NP_FNN_wb[@]}" "${commands_NP_FNN_sw_wb[@]}" "${commands_RNN_wb_in[@]}")

# initialize job counter
job_counter=1000

# loop through the python commands
for command in "${python_commands_wn_1[@]}"; do
    # create a job name based on the counter
    job_name="job_hsc_$job_counter"
    echo "Submitting $job_name"

    # create a temporary Slurm script
    echo "#!/bin/bash" > temp.sh
    echo "#SBATCH --job-name=$job_name" >> temp.sh
    echo "#SBATCH --output=res_$job_name" >> temp.sh

    # check if "Transformer" or "RNN" is in the command
    if [[ $command == *"Transformer"* ]] || [[ $command == *"RNN"* ]]; then
        echo "#SBATCH --time=03:30:00" >> temp.sh
        echo "#SBATCH -p gpu" >> temp.sh
        echo "#SBATCH -G 1" >> temp.sh
    else
        echo "#SBATCH --time=01:00:00" >> temp.sh
        echo "#SBATCH --cpus-per-task=10" >> temp.sh
        echo "#SBATCH --mem-per-cpu=1G" >> temp.sh
    fi

    # add the python command to the Slurm script
    echo $command >> temp.sh

    # submit the Slurm job
    sbatch temp.sh

    # remove the temporary Slurm script
    rm temp.sh

    # increment the job counter
    ((job_counter++))
done

pip uninstall -y darts
pip install git+https://github.com/LeonieFreisinger/darts.git@revba_nonlearnable#egg=darts

### RNN_wb_ba
commands_RNN_wb_ba=(
'python3 EXP_REAL_DATA.py --dataset EIA --model RNNModel --params RNN_wb_ba --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset London --model RNNModel --params RNN_wb_ba --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset ERCOT --model RNNModel --params RNN_wb_ba --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset Australian --model RNNModel --params RNN_wb_ba --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset Solar --model RNNModel --params RNN_wb_ba --gen_func "gen_model_and_params_none"'
'python3 EXP_REAL_DATA.py --dataset custom --data_path "../datasets/ETTh_panel.csv" --model RNNModel --params RNN_wb_ba --gen_func "gen_model_and_params_none"'
)

# combine the command lists
python_commands_wn_2=( "${commands_RNN_wb_ba[@]}")

# initialize job counter
job_counter=2000

# loop through the python commands
for command in "${python_commands_wn_2[@]}"; do
    # create a job name based on the counter
    job_name="job_hsc_$job_counter"
    echo "Submitting $job_name"

    # create a temporary Slurm script
    echo "#!/bin/bash" > temp.sh
    echo "#SBATCH --job-name=$job_name" >> temp.sh
    echo "#SBATCH --output=res_$job_name" >> temp.sh

    # check if "Transformer" or "RNN" is in the command
    if [[ $command == *"Transformer"* ]] || [[ $command == *"RNN"* ]]; then
        echo "#SBATCH --time=03:30:00" >> temp.sh
        echo "#SBATCH -p gpu" >> temp.sh
        echo "#SBATCH -G 1" >> temp.sh
    else
        echo "#SBATCH --time=01:00:00" >> temp.sh
        echo "#SBATCH --cpus-per-task=10" >> temp.sh
        echo "#SBATCH --mem-per-cpu=1G" >> temp.sh
    fi

    # add the python command to the Slurm script
    echo $command >> temp.sh

    # submit the Slurm job
    sbatch temp.sh

    # remove the temporary Slurm script
    rm temp.sh

    # increment the job counter
    ((job_counter++))
done