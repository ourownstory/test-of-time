#!/bin/bash

# load modules
module load python/3.9.0
module load cuda/11.7.1

# activate the virtual environment
source ../../tot4/bin/activate

# re-install the packages
pip uninstall -y neuralprophet
pip install git+https://github.com/ourownstory/neural_prophet.git@weighted-loss
pip install -e git+https://github.com/LeonieFreisinger/darts.git@lgbm_for_server#egg=darts

### NP
commands_NP=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'

)

### NP_localST
commands_NP_localST=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### NP_FNN
commands_NP_FNN=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NeuralProphetModel --params NP_FNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### TP
commands_TP=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### TP_localST
comannds_TP_localST=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TorchProphetModel --params TP_localST --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### TF
commands_TF=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model TransformerModel --params TF --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### RNN
commands_RNN=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model RNNModel --params RNN --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### LGBM
commands_LGBM=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model LightGBMModel --params LGBM --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### Naive
commands_Naive=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model NaiveModel --params Naive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

### SNaive
commands_SNaive=(
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,0"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "0,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_mean --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "0,0,2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "5,5" --data_offset_per_group "0,0" --data_amplitude_per_group "1,1" --proportion_break "2,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,1" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "2,1"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "1,10" --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "10,1"  --data_offset_per_group "10,1" --data_amplitude_per_group "10,1" --proportion_break "1,2"'
'python3 EXP_SYN_DATA.py --data_func gen_struc_break_var --model SeasonalNaiveModel --params SNaive --data_n_ts_groups "2,2,2,2" --data_offset_per_group "1,10,1,10" --data_amplitude_per_group "1,10,1,10" --proportion_break "1,1,2,2"'
)

# combine the command lists
python_commands=("${commands_NP[@]}" "${commands_NP_localST[@]}" "${commands_NP_FNN[@]}" "${commands_TP[@]}" "${commands_TP_localST[@]}" "${commands_NN[@]}" "${commands_RNN[@]}" "${commands_LGBM[@]}" "${commands_Naive[@]}" "${commands_SNaive[@]}")

# initialize job counter
job_counter=1

# loop through the python commands
for command in "${python_commands[@]}"; do
    # create a job name based on the counter
    job_name="job_sb_$job_counter"
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

