#!/bin/bash

#SBATCH --job-name=season_shape

#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=0.5G

### NP
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile1 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile2 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile3 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile4 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile5 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile6 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile7 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile8 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile9 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile10 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile11 &

### NP_localST
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile12 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile13 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile14 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile15 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile16 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile17 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile18 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile19 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile20 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile21 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile22 &

### NP_FNN
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile23 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile24 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile25 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile26 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile27 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile28 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile29 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile30 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile31 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile32 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile33 &

### TP
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile34 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile35 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile36 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile37 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile38 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile39 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile40 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile41 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile42 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile43 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile44 &

### TP_localST
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile45 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile46 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile47 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile48 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile49 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile50 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile51 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP_localST --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile52 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile53 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile54 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TorchProphetModel --params TP_localST --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile55 &

### TF
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile56 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TransformerModel --params TF --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model TransformerModel --params TF --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile63 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile66 &


### RNN
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile63 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile66 &


### LGBM
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile56 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model LightGBMModel --params LGBM --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model LightGBMModel --params LGBM --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile63 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile66 &

### Naive
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile56 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile63 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile66 &

### SNaive
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile56 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,2" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "+10,1" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile63 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,1,4,4" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "1000,1000,100,100"  --data_amplitude_per_group "50,5,50,5" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_canceling_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "4,4,1,1" --data_offset_per_group "1000,1000,100,100" --data_amplitude_per_group "50,5,50,5" > outfile66 &





