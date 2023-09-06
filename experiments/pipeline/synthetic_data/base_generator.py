import pandas as pd

from experiments.pipeline import synthetic_data


def generate(
    series_start,
    series_length,
    data_trend_gradient_per_group,
    data_func,
    n_ts_groups,
    offset_per_group,
    amplitude_per_group,
    proportion_break,
    freq,
):
    date_rng = pd.date_range(start=series_start, periods=series_length, freq=freq)
    if data_trend_gradient_per_group is not None:
        df = getattr(synthetic_data, data_func)(
            series_length=series_length,
            date_rng=date_rng,
            n_ts_groups=n_ts_groups,
            offset_per_group=offset_per_group,
            amplitude_per_group=amplitude_per_group,
            trend_gradient_per_group=data_trend_gradient_per_group,
        )
    elif proportion_break is not None:
        df = getattr(synthetic_data, data_func)(
            series_length=series_length,
            date_rng=date_rng,
            n_ts_groups=n_ts_groups,
            offset_per_group=offset_per_group,
            amplitude_per_group=amplitude_per_group,
            proportion_break=proportion_break,
        )
    else:
        df = getattr(synthetic_data, data_func)(
            series_length=series_length,
            date_rng=date_rng,
            n_ts_groups=n_ts_groups,
            offset_per_group=offset_per_group,
            amplitude_per_group=amplitude_per_group,
        )
    return df
