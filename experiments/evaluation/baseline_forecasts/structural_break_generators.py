import numpy as np
import pandas as pd
from scipy.stats import zscore

from statsmodels.tsa.arima_process import ArmaProcess

__all__ = ["gen_struc_break_mean", "gen_struc_break_var"]


def gen_struc_break_mean(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
    proportion_break: list,
) -> pd.DataFrame:
    df_data = []
    period = 24
    np.random.seed(42)

    proportion_season = 2
    proportion_ar = 1
    proportion_noise = 0.05

    t = np.arange(series_length)
    omega = 2 * np.pi / period
    ar_coeffs = np.array([1, 0.5, -0.1, 0.02, 0.3])
    ma_coeffs = np.array([1])  # MA coefficients (no MA component)
    ar_process = ArmaProcess(ar_coeffs, ma_coeffs, nobs=series_length)

    counter = 0
    for group_num in range(len(n_ts_groups)):  # looping over two groups
        n_ts = n_ts_groups[group_num]
        offset = offset_per_group[group_num]
        amplitude = amplitude_per_group[group_num]

        # generate season data with default scale 1
        season_group = [
            (np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t)) for _ in range(n_ts)
        ]
        # generate ar data with default scale 1
        ar_group = [ar_process.generate_sample(series_length) for _ in range(n_ts)]
        # generate noise data with default scale 1
        noise_group = [np.random.normal(loc=0, scale=1, size=series_length) for _ in range(n_ts)]

        # Define the step function
        step_function = np.zeros(series_length)
        step_function[series_length // 2 :] = (
            amplitude * proportion_break[group_num]
        )  # Adjust the step size relative to the amplitude

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset + step_function
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_struc_break_var(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
    proportion_break: list,
) -> pd.DataFrame:
    df_data = []
    period = 24
    np.random.seed(42)

    proportion_season = 2
    proportion_ar = 1
    proportion_noise = 0.05

    t = np.arange(series_length)
    omega = 2 * np.pi / period
    ar_coeffs = np.array([1, 0.5, -0.1, 0.02, 0.3])
    ma_coeffs = np.array([1])  # MA coefficients (no MA component)
    ar_process = ArmaProcess(ar_coeffs, ma_coeffs, nobs=series_length)

    counter = 0
    for group_num in range(len(n_ts_groups)):  # looping over two groups
        n_ts = n_ts_groups[group_num]
        offset = offset_per_group[group_num]
        amplitude = amplitude_per_group[group_num]

        # generate season data with default scale 1
        season_group = [
            (np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t)) for _ in range(n_ts)
        ]
        # generate ar data with default scale 1
        ar_group = [ar_process.generate_sample(series_length) for _ in range(n_ts)]
        # generate noise data with default scale 1
        noise_group = [np.random.normal(loc=0, scale=1, size=series_length) for _ in range(n_ts)]

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Change the variance in the second half of the series
            combined_data[series_length // 2 :] *= proportion_break[group_num]
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs