import logging
log = logging.getLogger("evaluation")


def get_model_params_list():
    return MODEL_PARAMS
def get_all_model_params_list():
    return ALL_MODEL_PARAMS
def get_model_groups():
    return MODEL_GROUPS
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

def get_baseline_experiments():
    return BASELINE_EPERIMENTS

def get_grouped_experiments():
    return GROUPED_EXPERIMENTS

MODEL_PARAMS = [
    "NP_localST",
    "NP_FNN_sw_wb",
    "NP_FNN_sw",
    "NP_FNN_wb",
    "NP_FNN",
    "NP",
    "TP_localST",
    "TP",
    "LGBM",
    "RNN_wb_in",
    "RNN_wb_ba",
    "RNN_wb",
    "RNN",
    "TF",
    # "SNaive",
    # "Naive",
]
ALL_MODEL_PARAMS = [
    "NP_localST",
    "NP_FNN_sw_wb",
    "NP_FNN_sw",
    "NP_FNN_wb",
    "NP_FNN",
    "NP",
    "TP_localST",
    "TP",
    "LGBM",
    "RNN_wb_in",
    "RNN_wb_ba",
    "RNN_wb",
    "RNN",
    "TF",
    "SNaive",
    "Naive",
]
MODEL_GROUPS = {
    "DECOMP": ["TP", "NP"],
    "DECOMP_LOCAL":["TP_localST", "NP_localST"],
    "DL":["NP_FNN","NP_FNN_sw","RNN"],
    "PARTLY_INT_WITH_NORM":["LGBM"],
    "DL_WITH_NORM":["TF"],
}
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
    'PowerTransformer()',
    # QuantileTransformer(output_distribution="normal"),
    'LogTransformer()',
]
DEFAULT_WEIGHTED = [
    'None',
    'std',
]

DEFAULT_SCALING_LEVEL = [
    'per_time_series',
    'per_dataset',
    'None'
]

DEFAULT_NORM_TYPE = [
    'instance',
    'batch',
    'None',
]

DEFAULT_NORM_AFFINE = [
    'True',
    'False',
]

BASELINE_EPERIMENTS = [
    "gen_one_shape_ar_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr",
    "gen_cancel_shape_ar_n_ts_[5, 5]_am_[10, 10]_of_[0, 0]_gr",
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]",
    "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]",
    "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]",
]
BASELINE_EPERIMENTS_WINDOW_BASED = [
    "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]",
    "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]",
    "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]",
]

ALL_MODELS_NON_WINDOW_BASED = ["NP_localST", "NP_FNN_sw", "NP_FNN_", "NP", "TP_localST", "TP", "LGBM", "RNN", "TF", "SNaive", "Naive"]
ALL_MODELS_WINDOW_BASED = ["NP_FNN_sw_wb", "NP_FNN_wb", "RNN_wb_in", "RNN_wb_ba"]
ALL_MODELS = ALL_MODELS_WINDOW_BASED + ALL_MODELS_NON_WINDOW_BASED


SEA_SUB_GROUPS = {
    "scale_amplitude_variations": {
        "gen_one_shape_ar_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr",
        "gen_one_shape_ar_n_ts_[5, 5]_am_[10, 1]_of_[0, 0]_gr",
        "gen_one_shape_ar_n_ts_[5, 5]_am_[10, 10]_of_[10, 1]_gr",
        "gen_one_shape_ar_n_ts_[2, 2, 2, 2]_am_[10, 10, 1, 1]_of_[1, 10, 1, 10]_gr",
    },
    "outliers": {
        "gen_one_shape_ar_outlier_0p1_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr",
        "gen_one_shape_ar_outlier_1p_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr"
    },
    "intermittent": {"generate_intermittent_n_ts_[5, 5]_am_[10, 1]_of_[0, 0]_gr"},
}

SEASH_SUB_GROUPS = {
    "scale_indep_pattern_balanced": [
        "gen_cancel_shape_ar_n_ts_[2, 2, 2, 2]_am_[1, 1, 1, 1]_of_[10, 10, 1, 1]_gr",
        "gen_cancel_shape_ar_n_ts_[1, 1, 2, 2]_am_[10, 1, 10, 1]_of_[10, 10, 1, 1]_gr",
        "gen_cancel_shape_ar_n_ts_[1, 1, 4, 4]_am_[10, 1, 10, 1]_of_[10, 10, 1, 1]_gr",
        "gen_cancel_shape_ar_n_ts_[4, 4, 1, 1]_am_[10, 1, 10, 1]_of_[10, 10, 1, 1]_gr",
        "gen_cancel_shape_ar_n_ts_[5, 5]_am_[10, 10]_of_[0, 0]_gr",
    ],
    "scale_dep_pattern_balanced": [
        "gen_cancel_shape_ar_n_ts_[2, 2, 2, 2]_am_[1, 1, 1, 1]_of_[10, 1, 10, 1]_gr",
        "gen_cancel_shape_ar_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr",
    ],
    "scale_dep_pattern_unbalanced_to_high": [
        "gen_cancel_shape_ar_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr",
    ],
    "scale_dep_pattern_unbalanced_to_low": [
        "gen_cancel_shape_ar_n_ts_[1, 2]_am_[10, 1]_of_[10, 1]_gr",
        "gen_cancel_shape_ar_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr",
    ],
}

TRE_SUB_GROUPS = {
    "scale_indep_trend_balanced": [
        "gen_one_shape_ar_trend_n_ts_[2, 2, 2, 2]_am_[10, 10, 1, 1]_of_[10, 10, 1, 1]_gr_[10.0, 0.0, 10.0, 0.0]",
        "gen_one_shape_ar_trend_n_ts_[1, 1, 2, 2]_am_[10, 10, 1, 1]_of_[10, 10, 1, 1]_gr_[1.0, 0.0, 1.0, 0.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[2, 2, 2, 2]_am_[10, 10, 1, 1]_of_[10, 10, 1, 1]_gr_[10.0, 0.0, 10.0, 0.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[1, 1, 2, 2]_am_[10, 10, 1, 1]_of_[10, 10, 1, 1]_gr_[1.0, 0.0, 1.0, 0.0]",
    ],
    "scale_dep_trend_on_off_unbalanced_to_low": [
        "gen_one_shape_ar_trend_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_[10.0, 0.0]",
        "gen_one_shape_ar_trend_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_[0.0, 1.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_[10.0, 0.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_[0.0, 1.0]",
    ],
    "scale_dep_trend_on_off_unbalanced_to_high": [
        "gen_one_shape_ar_trend_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_[10.0, 0.0]",
        "gen_one_shape_ar_trend_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_[0.0, 1.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_[10.0, 0.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_[0.0, 1.0]",
    ],
    "scale_dep_trend_balanced": [
        "gen_one_shape_ar_trend_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]",
        "gen_one_shape_ar_trend_cp_n_ts_[5, 5]_am_[10, 1]_of_[10, 1]_gr_[10.0, 1.0]",
    ],
}

HET_SUB_GROUPS = {
    "common_heteroscedacity": [
        "gen_one_shape_heteroscedacity_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]"
    ],
    "scale_dep_heteroscedacity_on_off_unbalanced_to_high": [
        "gen_one_shape_heteroscedacity_n_ts_[2, 1]_am_[10, 1]_of_[10, 1]_gr_[1.0, 0.0]",
        "gen_one_shape_heteroscedacity_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_[0.0, 1.0]",
        "gen_one_shape_heteroscedacity_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_[1.0, 0.0]",
    ],
    "scale_dep_heteroscedacity_on_off_unbalanced_to_low": [
        "gen_one_shape_heteroscedacity_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_[0.0, 1.0]",
        "gen_one_shape_heteroscedacity_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_[1.0, 0.0]",
    ],
    "opposing_heteroscedacity": [
        "gen_one_shape_heteroscedacity_op_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]",
        "gen_one_shape_heteroscedacity_op_n_ts_[10, 1]_am_[1, 1]_of_[0, 0]_gr_[1.0, 1.0]",
    ],
}
STRUC_SUB_GROUPS = {
    "common_structural_break":[
        "gen_struc_break_mean_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]",
        "gen_struc_break_var_n_ts_[5, 5]_am_[1, 1]_of_[0, 0]_gr_None_[2, 2]",
    ],
    "scale_dep_structural_break_on_off_unbalanced_to_high": [
        "gen_struc_break_mean_n_ts_[2, 1]_am_[10, 1]_of_[10, 1]_gr_None_[2, 0]",
        "gen_struc_break_mean_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_None_[0, 2]",
        "gen_struc_break_mean_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_None_[2, 0]",
        "gen_struc_break_var_n_ts_[2, 1]_am_[10, 1]_of_[10, 1]_gr_None_[2, 2]",
        "gen_struc_break_var_n_ts_[2, 1]_am_[10, 1]_of_[10, 1]_gr_None_[2, 1]",
        "gen_struc_break_var_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_None_[1, 2]",
        "gen_struc_break_var_n_ts_[10, 1]_am_[10, 1]_of_[10, 1]_gr_None_[2, 1]",
    ],
    "scale_dep_structural_break_on_off_unbalanced_to_low": [
        "gen_struc_break_mean_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_None_[0, 2]",
        "gen_struc_break_mean_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_None_[2, 0]",
        "gen_struc_break_var_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_None_[1, 2]",
        "gen_struc_break_var_n_ts_[1, 10]_am_[10, 1]_of_[10, 1]_gr_None_[2, 1]",

    ],
    "scale_indep_structural_break": [
        "gen_struc_break_mean_n_ts_[2, 2, 2, 2]_am_[1, 10, 1, 10]_of_[1, 10, 1, 10]_gr_None_[0, 0, 2, 2]",
        "gen_struc_break_var_n_ts_[2, 2, 2, 2]_am_[1, 10, 1, 10]_of_[1, 10, 1, 10]_gr_None_[1, 1, 2, 2]",
    ],
}

GROUPED_EXPERIMENTS = {
    "SEA_SUB_GROUPS": SEA_SUB_GROUPS,
    "SEASH_SUB_GROUPS": SEASH_SUB_GROUPS,
    "TRE_SUB_GROUPS": TRE_SUB_GROUPS,
    "STRUC_SUB_GROUPS": STRUC_SUB_GROUPS,
    "HET_SUB_GROUPS": HET_SUB_GROUPS,
}