import numpy as np
from experiments.evaluation.baseline_forecasts.one_shape_generators import gen_one_shape_ar
import pandas as pd
from experiments.evaluation.baseline_forecasts.metric_caluclator import calc_per_series_metrics
from experiments.evaluation.baseline_forecasts.exp_params import get_exp_params

SERIES_LENGTH = 24 * 1
DATE_RNG = pd.date_range(start=pd.to_datetime("2011-01-01 01:00:00"), periods=SERIES_LENGTH, freq="H")
freq = "H"
series_length = 24 * 7 * 15
series_start = pd.to_datetime("2011-01-01 01:00:00")

synthetic_data_dfs = []
# df = gen_one_shape_ar(
#         series_length=SERIES_LENGTH,
#         date_rng=DATE_RNG,
#         n_ts_groups=[1,1],
#         offset_per_group=[1,10],
#         amplitude_per_group=[1,10],
#         calc_without_noise=True,
# )
df = generate(
        series_start=series_start,
        series_length=series_length,
        data_trend_gradient_per_group=data_trend_gradient_per_group,
        data_func=args.data_func,
        n_ts_groups=data_n_ts_groups,
        offset_per_group=data_offset_per_group,
        amplitude_per_group=data_amplitude_per_group,
        proportion_break=proportion_break,
        freq=freq,
    )
for df in synthetic_data_dfs:
    df_metric = calc_per_series_metrics(df, ['MAE', 'RMSE'])
print(df_metric)
