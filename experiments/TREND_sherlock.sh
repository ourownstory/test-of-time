#!/bin/bash

#SBATCH --job-name=trend
#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH --cpus-per-task=19
#SBATCH --mem-per-cpu=1G
#SBATCH --output=myjob-%A_%a.out


### NP
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### NP_localST
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### NP_FNN
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### TP
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### TP_localST
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### TF
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### RNN
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### LGBM
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### Naive
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"

### SNaive
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"
 python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0"





