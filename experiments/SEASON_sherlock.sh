#!/bin/bash

#SBATCH --job-name=test

#SBATCH --time=2:00:00
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

### NP
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile1 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile2 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile3 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile4 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile5 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile6 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile7 &

### NP_localST
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile8 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile9 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile10 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile11 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile12 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile13 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile14 &

### NP_FNN
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile15 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile16 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile17 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile18 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile19 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile20 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile21 &

### TP
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile22 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile23 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile24 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile25 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile26 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile27 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile28 &

### TP_localST
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile29 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile30 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile31 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile32 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile33 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile34 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile35 &

### TF
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile36 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile37 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile38 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile39 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile40 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile41 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile42 &

### RNN
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile43 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile44 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile45 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile46 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile47 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile48 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile49 &

### LGBM
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile50 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile51 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile52 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile53 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile54 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile55 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile56 &

### Naive
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile57 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile58 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile59 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile60 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile61 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile62 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile63 &

### SNaive
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" > outfile64 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile65 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" > outfile66 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile67 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" > outfile68 &
nohup python3 EXP_SYN_DATA.py --data_func generate_intermittent --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" > outfile69 &
nohup python3 EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" > outfile70 &












