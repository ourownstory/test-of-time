#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# ### NP
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile1 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile2 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile3 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile4 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile5 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile6 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile7 &

# ### NP_localST
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile8 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile9 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile10 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile11 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile12 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile13 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile14 &

# ### NP_FNN
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile15 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile16 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile17 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile18 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile19 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile20 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile21 &

# ### TP
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile22 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile23 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile24 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile25 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile26 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile27 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile28 &

# ### TP_localST
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile29 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile30 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile31 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile32 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile33 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile34 &
# nohup python3 vEXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile35 &

# ### TF
 CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile36 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile37 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile38 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile39 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile40 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile41 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile42 &

# # # ### RNN
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile43 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile44 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile45 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile46 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile47 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile48 &
# CUDA_LAUNCH_BLOCKING=1 nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile49 &

# ### LGBM
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile50 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile51 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile52 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile53 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile54 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile55 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile56 &

# ### Naive
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile57 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile58 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile59 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile60 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile61 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile62 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile63 &

# ### SNaive
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > soutfile64 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile65 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > soutfile66 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile67 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > soutfile68 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > soutfile69 &
# nohup python3 experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > soutfile70 &
