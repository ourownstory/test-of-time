import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

__all__ = [
    "gen_one_shape_ar",
    "gen_one_shape_ar_outlier_0p1",
    "gen_one_shape_ar_outlier_1p",
    "gen_one_shape_ar_trend",
    "gen_one_shape_ar_trend_cp",
    "gen_one_shape_heteroscedacity",
    "gen_one_shape_heteroscedacity_op",
]


def generate_one_shape_season_data(  # OUTDATED
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
) -> pd.DataFrame:
    df_seasons = []
    period = 24
    np.random.seed(42)
    t = np.arange(series_length)  # Generate an array of time steps
    omega = 2 * np.pi / period  # Define the angular frequency (omega) corresponding to the period
    ar_coeffs = np.array([1, 0.5, -0.1, 0.02, 0.3])  # Define AR coefficients (AR(4) model)
    ma_coeffs = np.array([1])  # MA coefficients (no MA component)
    ar_process = ArmaProcess(ar_coeffs, ma_coeffs, nobs=series_length)  # Create an ARMA process

    counter = 0
    for group in range(len(n_ts_groups)):
        noise_group = [
            np.random.normal(loc=0, scale=amplitude_per_group[group] / 5, size=series_length)
            for _ in range(n_ts_groups[group])
        ]
        data_group = [
            (np.sin(omega * t) + np.cos(omega * t) + np.sin(2 * omega * t) + np.cos(2 * omega * t))
            * amplitude_per_group[group]
            for _ in range(n_ts_groups[group])
        ]
        for i in range(n_ts_groups[group]):
            df = pd.DataFrame(date_rng, columns=["ds"])
            df["y"] = data_group[i] + noise_group[i] + offset_per_group[group]
            df["ID"] = str(counter)
            df_seasons.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_seasons, axis=0)
    return concatenated_dfs


def gen_one_shape_ar(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
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
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_one_shape_ar_outlier_0p1(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
) -> pd.DataFrame:
    df_data = []
    period = 24
    np.random.seed(42)

    proportion_season = 2
    proportion_ar = 1
    proportion_noise = 0.05
    proportion_outliers = 0.5
    num_outliers = int(round(series_length * 0.1))  # 0,1% outliers

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

        outlier_positions = [
            np.random.choice(range(series_length), size=num_outliers, replace=False)
            for i in range(n_ts_groups[group_num])
        ]
        outlier_group = [np.random.normal(loc=0, scale=1, size=num_outliers) for _ in range(n_ts_groups[group_num])]

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Add the outliers at the specified positions
            combined_data[outlier_positions[i]] += proportion_outliers * outlier_group[i]
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

        concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_one_shape_ar_outlier_1p(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
) -> pd.DataFrame:
    df_data = []
    period = 24
    np.random.seed(42)

    proportion_season = 2
    proportion_ar = 1
    proportion_noise = 0.05
    proportion_outliers = 0.5
    num_outliers = int(round(series_length * 1.0))  # 1% outliers

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

        outlier_positions = [
            np.random.choice(range(series_length), size=num_outliers, replace=False)
            for i in range(n_ts_groups[group_num])
        ]
        outlier_group = [np.random.normal(loc=0, scale=1, size=num_outliers) for _ in range(n_ts_groups[group_num])]

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Add the outliers at the specified positions
            combined_data[outlier_positions[i]] += proportion_outliers * outlier_group[i]
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

        concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_one_shape_ar_trend(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
    trend_gradient_per_group: list,
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
        # Generate the trend for the group
        trend = np.linspace(
            0, trend_gradient_per_group[group_num] * 15, series_length
        )  # sets the trend gradient to a proper range

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset + trend
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_one_shape_ar_trend_cp(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
    trend_gradient_per_group: list,
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
        # Generate the trend for the group
        trend = np.concatenate(
            [
                np.linspace(
                    0, trend_gradient_per_group[group_num] * 15 / 2, series_length // 2
                ),  # same gradient as without changepoint
                np.linspace(
                    trend_gradient_per_group[group_num] * 15 / 2, 0, series_length // 2
                ),  # same gradient as without changepoint
            ]
        )

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + offset + trend
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_one_shape_heteroscedacity(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
    trend_gradient_per_group: list,
) -> pd.DataFrame:
    df_data = []
    period = 24
    np.random.seed(42)

    proportion_season = 2
    proportion_ar = 1
    proportion_noise = 0.05
    proportion_var_trend = 0.05

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
        # Generate the trend for the group
        trend = np.linspace(
            0, trend_gradient_per_group[group_num] * 5, series_length
        )  # sets the trend gradient to a proper range

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]

            # Add trend-dependent variance and scale with z-score, only if trend is not zero
            if np.any(trend != 0):
                combined_data = combined_data * np.sqrt(trend) * proportion_var_trend
                mean = np.mean(combined_data)
                std_dev = np.std(combined_data)
                combined_data = (combined_data - mean) / std_dev

            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + trend + offset
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs


def gen_one_shape_heteroscedacity_op(
    series_length: int,
    date_rng,
    n_ts_groups: list,
    offset_per_group: list,
    amplitude_per_group: list,
    trend_gradient_per_group: list,
) -> pd.DataFrame:
    df_data = []
    period = 24
    np.random.seed(42)

    proportion_season = 2
    proportion_ar = 1
    proportion_noise = 0.05
    proportion_var_trend = 0.1

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
        # Generate the trend for the group
        trend = np.linspace(
            0, trend_gradient_per_group[group_num] * 0.5, series_length
        )  # sets the trend gradient to a proper range

        # If the group number is even, make the trend negative and decrease the variance
        if group_num % 2 == 0:
            trend = trend[::-1]

        for i in range(n_ts):
            df = pd.DataFrame(date_rng, columns=["ds"])
            # Add the season_group and ar_group according to desired proportion, then scale them individually with z-score
            combined_data = proportion_season * season_group[i] + proportion_ar * ar_group[i]
            mean = np.mean(combined_data)
            std_dev = np.std(combined_data)
            combined_data = (combined_data - mean) / std_dev
            # Add the noise according to desired proportion
            combined_data += proportion_noise * noise_group[i]
            # Add trend-dependent variance and scale with z-score
            combined_data = combined_data * ((-1) ** group_num) * np.sqrt(abs(trend)) * proportion_var_trend
            # Scale the series with the amplitude and offset of the respective group
            df["y"] = combined_data * amplitude + trend + offset
            df["ID"] = str(counter)
            df_data.append(df.reset_index(drop=True))
            counter += 1

    concatenated_dfs = pd.concat(df_data, axis=0)

    # Apply global normalization to ensure standardized variance for dataset
    concatenated_dfs["y"] = (concatenated_dfs["y"]) / concatenated_dfs["y"].std()

    return concatenated_dfs
