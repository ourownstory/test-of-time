#!/bin/bash

#!/bin/bash

# load modules
#module load python/3.9.0
#module load cuda/11.7.1

# activate the virtual environment
#source ../../tot4/bin/activate

# re-install the packages
pip uninstall -y neuralprophet
pip install git+https://github.com/ourownstory/neural_prophet.git@instance-batch-normalization
pip install -e git+https://github.com/LeonieFreisinger/darts.git@revin_learnable#egg=darts

#git checkout evolution_experiments_1
#git pull upstream evolution_experiments_1

nohup python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" --gen_func gen_model_and_params_norm > outfile1
nohup python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" --gen_func gen_model_and_params_norm > outfile2


### NP_FNN SEASON
commands_NP_FNN_SEASON=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" --gen_func gen_model_and_params_norm'
)

### NP_FNN SEASHAPE
commands_NP_FNN_SEASHAPE=(
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" --gen_func gen_model_and_params_norm'
)

### NP_FNN TREND
commands_NP_FNN_TREND=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" --gen_func gen_model_and_params_norm '
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group 0"0.3,0,0.03,0" --gen_func gen_model_and_params_norm'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" --gen_func gen_model_and_params_norm'
)



### RNN SEASON
commands_RNN_SEASON=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" --gen_func gen_model_and_params_none'
)

### RNN SEASHAPE
commands_RNN_SEASHAPE=(
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN_wb --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN_wb --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" --gen_func gen_model_and_params_none'
)

### RNN TREND
commands_RNN_TREND=(
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" --gen_func gen_model_and_params_none'
'python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" --gen_func gen_model_and_params_none'
)


# combine the command lists
python_commands=("${commands_NP_FNN_SEASON[@]}" "${commands_NP_FNN_SEASHAPE[@]}" "${commands_NP_FNN_TREND[@]}" "${commands_RNN_SEASON[@]}" "${commands_RNN_SEASHAPE[@]}" "${commands_RNN_TREND[@]}" )

# initialize job counter
job_counter=1

# loop through the python commands
for command in "${python_commands[@]}"; do
    # create a job name based on the counter
    job_name="job_wn_$job_counter"
    echo "Submitting $job_name"

    # create a temporary Slurm script
    echo "#!/bin/bash" > temp.sh
    echo "#SBATCH --job-name=$job_name" >> temp.sh
    echo "#SBATCH --output=res_$job_name" >> temp.sh

     # check if "Transformer" or "RNN" is in the command
    if [[ $command == *"Transformer"* ]] || [[ $command == *"RNN"* ]]; then
        echo "#SBATCH --time=02:00:00" >> temp.sh
        echo "#SBATCH -p gpu" >> temp.sh
        echo "#SBATCH -G 1" >> temp.sh
    else
        echo "#SBATCH --time=00:20:00" >> temp.sh
        echo "#SBATCH --cpus-per-task=19" >> temp.sh
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




