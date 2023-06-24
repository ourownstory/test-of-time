import numpy as np


def calc_ar_series(ar_group, coeffs):
    ar_group_padded = np.pad(ar_group, ((0, 0), (0, 4)), mode='constant', constant_values=0)
    coeffs = np.flip(coeffs)
    ar_group_predictions = np.apply_along_axis(lambda row: np.convolve(row, coeffs, mode='valid'), axis=1, arr=ar_group_padded)
    return ar_group_predictions

