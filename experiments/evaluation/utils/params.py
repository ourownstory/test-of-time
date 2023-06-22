import logging
log = logging.getLogger("evaluation")


def get_model_params_list():
    return MODEL_PARAMS
def get_data_group_keyword():
    return DATA_GROUP_KEYWORD
def get_rnn_norm_type_keyword():
    return RNN_NORM_TYPE_KEYWORD
def get_window_based_model_name_keyword():
    return WINDOW_BASED_MODEL_NAMES_KEYWORD

def get_default_scaler():
    return DEFAULT_SCALER

def get_default_weighted():
    return DEFAULT_WEIGHTED

def get_default_scaling_level():
    return DEFAULT_SCALING_LEVEL

def get_default_norm_type():
    return DEFAULT_NORM_TYPE

def get_default_norm_affine():
    return DEFAULT_NORM_AFFINE

MODEL_PARAMS = [
    "NP_localST_",
    "NP_FNN_sw_wb_",
    "NP_FNN_sw_",
    "NP_FNN_wb_",
    "NP_FNN_",
    "NP_",
    "TP_localST_",
    "TP_",
    "LGBM_",
    "RNN_wb_in_",
    "RNN_wb_ba_",
    "RNN_wb_",
    "RNN_",
    "TF_",
    "SNaive_",
    "Naive_",
]

DATA_GROUP_KEYWORD = {
    "_trend_": "TRE",
    "hetero": "HET",
    "struc_break": "STRU",
}

RNN_NORM_TYPE_KEYWORD = {
    "RNN_wb_in": "instance",
    "RNN_wb_ba": "batch",
}

WINDOW_BASED_MODEL_NAMES_KEYWORD = {
    "RNN_wb_in": "RNN",
    "RNN_wb_ba": "RNN",
    "NP_FNN_sw_wb": "NP_FNN_sw",
    "NP_FNN_wb": "NP_FNN",
}

DEFAULT_SCALER = [
    'None',
    'StandardScaler()',
    # 'MinMaxScaler(feature_range=(-0.5, 0.5))',
    'MinMaxScaler()',
    'RobustScaler(quantile_range=(5, 95))',
    # ShiftedBoxCoxTransformer(),
    'PowerTransformer(method="yeo-johnson", standardize=True)',
    # QuantileTransformer(output_distribution="normal"),
   ' LogTransformer()',
]
DEFAULT_WEIGHTED = [
    'None',
    'std',
]

DEFAULT_SCALING_LEVEL = [
    'per_time_series',
    'per_dataset',
]

DEFAULT_NORM_TYPE = [
    'instance',
    'batch',
    'None'
]

DEFAULT_NORM_AFFINE = [
    'True',
    'False',
]

