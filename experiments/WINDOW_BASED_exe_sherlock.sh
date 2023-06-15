#!/bin/bash

# load modules
module load python/3.9.0
module load cuda/11.7.1

# activate the virtual environment
source ../../tot4/bin/activate

# re-install the packages
pip uninstall -y neuralprophet
pip install git+https://github.com/ourownstory/neural_prophet.git@instance-batch-normalization
pip install -e git+https://github.com/LeonieFreisinger/darts.git@revin_learnable#egg=darts
#
##git checkout evolution_experiments_1
##git pull upstream evolution_experiments_1


### NP_FNN SEASON
commands_NP_FNN_SEASON=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "10,10,1,1" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN SEASHAPE
commands_NP_FNN_SEASHAPE=(
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "+10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,4,4" --data_offset_per_group "10,10,1,1" --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10,1,1"  --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "4,4,1,1" --data_offset_per_group "10,10,1,1" --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN TREND
commands_NP_FNN_TREND=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN STRUCBREAK
commands_NP_FNN_STRUCBREAK=(
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN HETEROSC
commands_NP_FNN_HETEROSC=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity_op --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity_op --model NeuralProphetModel --params NP_FNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN_SW SEASON
commands_NP_FNN_SW_SEASON=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "10,10,1,1" --gen_func "gen_model_and_params_norm"'
)
#
### NP_FNN_SW SEASHAPE
commands_NP_FNN_SW_SEASHAPE=(
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "+10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,1,4,4" --data_offset_per_group "10,10,1,1" --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10,1,1"  --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "4,4,1,1" --data_offset_per_group "10,10,1,1" --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN_SW TREND
commands_NP_FNN_SW_TREND=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN_SW STRUCBREAK
commands_NP_FNN_SW_STRUCBREAK=(
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2" --gen_func "gen_model_and_params_norm"'
)

### NP_FNN_SW HETEROSC
commands_NP_FNN_SW_HETEROSC=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_norm"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_norm "'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity_op --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_norm"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity_op --model NeuralProphetModel --params NP_FNN_sw_wb --data_n_ts_groups "10,1" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_norm"'
)



### RNN SEASON
commands_RNN_SEASON=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_outlier_0p1 --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_outlier_1p --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func generate_intermittent --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "10,10,1,1" --gen_func "gen_model_and_params_none"'
)
#
### RNN SEASHAPE
commands_RNN_SEASHAPE=(
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "10,10" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model RNNModel --params RNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_0p1 --model RNNModel --params RNN_wb --data_n_ts_groups "+10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_non"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model RNNModel --params RNN_wb --data_n_ts_groups "1,2" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar_outlier_1p --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,4,4" --data_offset_per_group "10,10,1,1" --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10,1,1"  --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_cancel_shape_ar --model RNNModel --params RNN_wb --data_n_ts_groups "4,4,1,1" --data_offset_per_group "10,10,1,1" --data_amplitude_per_group "10,1,10,1" --gen_func "gen_model_and_params_none"'
)

### RNN TREND
commands_RNN_TREND=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN_wb --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_none"'
)

### RNN STRUCBREAK
commands_RNN_STRUCBREAK=(
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN_wb --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2" --gen_func "gen_model_and_params_none"'
)

### RNN HETEROSC
commands_RNN_HETEROSC=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model RNNModel --params RNN_wb --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "1,0" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model RNNModel --params RNN_wb --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity_op --model RNNModel --params RNN_wb --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_none"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_heteroscedacity_op --model RNNModel --params RNN_wb --data_n_ts_groups "10,1" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --data_trend_gradient_per_group "1,1" --gen_func "gen_model_and_params_none"'
)



#combine the command lists
python_commands=("${commands_NP_FNN_SEASON[@]}" "${commands_NP_FNN_SEASHAPE[@]}" "${commands_NP_FNN_TREND[@]}" "${commands_NP_FNN_STRUCBREAK[@]}" "${commands_NP_FNN_HETEROSC[@]}" "${commands_NP_FNN_SW_SEASON[@]}" "${commands_NP_FNN_SW_SEASHAPE[@]}" "${commands_NP_FNN_SW_TREND[@]}" "${commands_NP_FNN_SW_STRUCBREAK[@]}" "${commands_NP_FNN_SW_HETEROSC[@]}" "${commands_RNN_SEASON[@]}" "${commands_RNN_SEASHAPE[@]}" "${commands_RNN_TREND[@]}" "${commands_RNN_HETEROSC[@]}" "${commands_RNN_STRUCBREAK[@]}")

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
        echo "#SBATCH --time=00:40:00" >> temp.sh
        echo "#SBATCH -p gpu" >> temp.sh
        echo "#SBATCH -G 1" >> temp.sh
    else
        echo "#SBATCH --time=00:35:00" >> temp.sh
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

#
#pip install -e git+https://github.com/LeonieFreisinger/darts.git@revin_nonlearnable#egg=darts
#
##combine the command lists
#python_commands_2=("${commands_RNN_SEASON[@]}" "${commands_RNN_SEASHAPE[@]}" "${commands_RNN_TREND[@]}" "${commands_RNN_HETEROSC[@]}" "${commands_RNN_STRUCBREAK[@]}")
#
## initialize job counter
#job_counter=1000
#
## loop through the python commands
#for command in "${python_commands_2[@]}"; do
#    # create a job name based on the counter
#    job_name="job_wn_$job_counter"
#    echo "Submitting $job_name"
#
#    # create a temporary Slurm script
#    echo "#!/bin/bash" > temp.sh
#    echo "#SBATCH --job-name=$job_name" >> temp.sh
#    echo "#SBATCH --output=res_$job_name" >> temp.sh
#
#    # check if "Transformer" or "RNN" is in the command
#    if [[ $command == *"Transformer"* ]] || [[ $command == *"RNN"* ]]; then
#        echo "#SBATCH --time=02:00:00" >> temp.sh
#        echo "#SBATCH -p gpu" >> temp.sh
#        echo "#SBATCH -G 1" >> temp.sh
#    else
#        echo "#SBATCH --time=00:30:00" >> temp.sh
#        echo "#SBATCH --cpus-per-task=19" >> temp.sh
#        echo "#SBATCH --mem-per-cpu=1G" >> temp.sh
#    fi
#
#    # add the python command to the Slurm script
#    echo $command >> temp.sh
#
#    # submit the Slurm job
#    sbatch temp.sh
#
#    # remove the temporary Slurm script
#    rm temp.sh
#
#    # increment the job counter
#    ((job_counter++))
#done
