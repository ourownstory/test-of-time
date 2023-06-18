import numpy as np
import pandas as pd
from plotly_resampler import unregister_plotly_resampler

__all__ = ["generate_intermittent"]


def generate_intermittent(
    series_length: int, date_rng, n_ts_groups: list, amplitude_per_group: list, offset_per_group: list = []
) -> pd.DataFrame:
    np.random.seed(42)

    # Define the proportion of non-zero values
    proportion_non_zeros = 0.5
    proportion_noise = 0.1

    # Define the length of the time series
    hours_per_day = 24
    days = series_length // hours_per_day

    # Initialize an empty DataFrame to hold all time series
    all_series_df = pd.DataFrame()

    # Initialize a counter for the ID
    id_counter = 0

    # Define the start hour for non-zero values
    start_hour = np.random.choice(range(hours_per_day - int(hours_per_day * proportion_non_zeros)))
    end_hour = start_hour + int(hours_per_day * proportion_non_zeros)

    # Generate a common daily pattern for all groups
    common_daily_pattern = np.zeros(hours_per_day)
    window = np.hanning(end_hour - start_hour)
    common_daily_pattern[start_hour:end_hour] = (
        np.random.exponential(scale=1, size=end_hour - start_hour) + 0.5  # * window
    )

    # Apply min-max normalization to the common daily pattern
    common_daily_pattern = (common_daily_pattern - common_daily_pattern.min()) / (
        common_daily_pattern.max() - common_daily_pattern.min()
    )

    # Generate time series for each group
    for group_idx, n_ts in enumerate(n_ts_groups):
        # Scale the common daily pattern by the group amplitude
        daily_pattern = common_daily_pattern

        for ts_idx in range(n_ts):
            # Add noise to the non-zero values to create the time series
            data = np.tile(daily_pattern, days)
            data[data != 0] += proportion_noise * np.random.normal(loc=0, scale=1, size=len(data[data != 0]))

            # Scale the series with the amplitude and offset of the respective group
            data[data != 0] = data[data != 0] * amplitude_per_group[group_idx] * 2 + offset_per_group[group_idx]

            # Create a DataFrame for this time series
            ts_df = pd.DataFrame()
            ts_df["ds"] = date_rng[: len(data)]
            ts_df["y"] = data
            ts_df["ID"] = id_counter

            # Append this time series to the overall DataFrame
            all_series_df = pd.concat([all_series_df, ts_df], ignore_index=True)

            # Increment the ID counter
            id_counter += 1
    # Apply global normalization to ensure standardized variance for dataset
    all_series_df["y"] = (all_series_df["y"]) / all_series_df["y"].std()

    return all_series_df
