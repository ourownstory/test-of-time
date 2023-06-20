#!/bin/bash

# load modules
module load python/3.9.0
module load cuda/11.7.1

# activate the virtual environment
source ../../tot4/bin/activate

# re-install the packages
pip uninstall -y neuralprophet
pip install git+https://github.com/ourownstory/neural_prophet.git@normalization-layer
pip uninstall -y darts
pip install git+https://github.com/LeonieFreisinger/darts.git@lgbm_for_server#egg=darts


### NP
commands_NP=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers_reweighting"'
)

### NP_localST
commands_NP_localST=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers_reweighting"'
)

### NP_FNN
commands_NP_FNN=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1"--data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers_reweighting"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers_reweighting"'
)

### TP
commands_TP=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

### TP_localST
commands_TP_localST=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1"  --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

### TF
commands_TF=(
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model TransformerModel --params TF --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model TransformerModel --params TF --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

### RNN
commands_RNN=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model RNNModel --params RNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model RNNModel --params RNN --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

### LGBM
commands_LGBM=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model LightGBMModel --params LGBM --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model LightGBMModel --params LGBM --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

### Naive
commands_Naive=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model NaiveModel --params Naive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model NaiveModel --params Naive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

### SNaive
commands_SNaive=(
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --data_trend_gradient_per_group "0,1" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "10,0,10,0" --gen_func "gen_model_and_params_scalers"'
#'python3 EXP_SYN_DATA.py --data_func gen_one_shape_ar_trend_cp --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,1,2,2" --data_offset_per_group "10,10, 1, 1" --data_amplitude_per_group "10,10,1,1" --data_trend_gradient_per_group "1,0,1,0" --gen_func "gen_model_and_params_scalers"'
)

# combine the command lists
python_commands=("${commands_NP[@]}" "${commands_NP_localST[@]}" "${commands_NP_FNN[@]}" "${commands_TP[@]}" "${commands_TP_localST[@]}" "${commands_TF[@]}" "${commands_RNN[@]}" "${commands_LGBM[@]}" "${commands_Naive[@]}" "${commands_SNaive[@]}")

# initialize job counter
job_counter=1

# loop through the python commands
for command in "${python_commands[@]}"; do
    # create a job name based on the counter
    job_name="job_t_$job_counter"
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
        echo "#SBATCH --time=00:20:00" >> temp.sh
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

