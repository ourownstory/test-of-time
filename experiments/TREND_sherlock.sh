#!/bin/bash

#SBATCH --job-name=trend

#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=0.5G


### NP
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile1 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile2 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile3 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile4 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile5 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile6 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile7 &

### NP_localST
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile8 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile9 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile10 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile11 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile12 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile13 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile14 &

### NP_FNN
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile15 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile16 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile17 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile18 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile19 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile20 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile21 &

### TP
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile22 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile23 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile24 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile25 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile26 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile27 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile28 &

### TP_localST
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile29 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile30 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile31 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile32 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile33 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile34 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile35 &

### TF
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile36 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile37 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile38 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile39 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile40 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile41 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model TransformerModel --params TF --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile42 &

### RNN
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile43 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile44 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile45 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile46 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile47 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile48 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model RNNModel --params RNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile49 &

### LGBM
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile50 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile51 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile52 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile53 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile54 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile55 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile56 &

### Naive
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model NaiveModel --params Naive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile63 &

### SNaive
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0.03" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0.3,0" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" --data_trend_gradient_per_group "0.3,0" > outfile66 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile67 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" --data_trend_gradient_per_group "0,0.03" > outfile68 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile69 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_and_trend_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000, 100, 100" --data_amplitude_per_group "50,50,5,5" --data_trend_gradient_per_group "0.3,0,0.03,0" > outfile70 &





