#!/bin/bash

### NP
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### NP_localST
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### NP_FNN
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### TP
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### TP_localST
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NeuralProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### TF
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model DartsForecastingModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### RNN
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model DartsForecastingModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### LGBM
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model DartsForecastingModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model DartsForecastingModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### Naive
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &

### SNaive
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,50" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data_with_outlier --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "1000,100" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_intermittent --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "50,5" &
python experiments/EXP_SYN_DATA.py --data_func generate_one_shape_season_and_ar_data --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "100,1000,100,1000" --data_amplitude_per_group "50,50,5,5" &












