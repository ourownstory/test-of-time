import logging
import numpy as np

log = logging.getLogger("tot.metric")


def _calc_mae(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAE error."""
    error_abs = np.abs(np.subtract(truth, predictions))
    return 1.0 * np.nanmean(error_abs, dtype="float32")


def _calc_mse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MSE error."""
    error_squared = np.square(np.subtract(truth, predictions))
    return 1.0 * np.nanmean(error_squared, dtype="float32")


def _calc_rmse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates RMSE error."""
    mse = _calc_mse(predictions, truth)
    return np.sqrt(mse)


def _calc_mase(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray,
) -> float:
    """Calculates MASE error.
    according to https://robjhyndman.com/papers/mase.pdf
    Note: Naive error is computed over in-sample data.
        MASE = MAE / NaiveMAE,
    where: MAE = mean(|actual - forecast|)
    where: NaiveMAE = mean(|actual_[i] - actual_[i-1]|)
    """
    assert len(truth_train) > 1
    mae = _calc_mae(predictions, truth)
    naive_mae = _calc_mae(np.array(truth_train[:-1]), np.array(truth_train[1:]))
    return np.divide(mae, 1e-9 + naive_mae)


def _calc_rmsse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray,
) -> float:
    """Calculates RMSSE error.
    according to https://robjhyndman.com/papers/mase.pdf
    Note: Naive error is computed over in-sample data.
    MSSE = RMSE / NaiveRMSE,
    where: RMSE = sqrt(mean((actual - forecast)^2))
    where: NaiveMSE = sqrt(mean((actual_[i] - actual_[i-1])^2))
    """
    assert len(truth_train) > 1
    rmse = _calc_rmse(predictions, truth)
    naive_rmse = _calc_rmse(np.array(truth_train[:-1]), np.array(truth_train[1:]))
    return np.divide(rmse, 1e-9 + naive_rmse)


def _calc_mape(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAPE error."""
    error = np.subtract(truth, predictions)
    error_relative = np.abs(np.divide(error, truth))
    return 100.0 * np.nanmean(error_relative, dtype="float32")


def _calc_smape(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates SMAPE error."""
    absolute_error = np.abs(np.subtract(truth, predictions))
    absolute_sum = np.abs(truth) + np.abs(predictions)
    error_relative_sym = np.divide(absolute_error, absolute_sum)
    return 100.0 * np.nanmean(error_relative_sym, dtype="float32")


ERROR_FUNCTIONS = {
    "MAE": _calc_mae,
    "MSE": _calc_mse,
    "RMSE": _calc_rmse,
    "MASE": _calc_mase,
    "RMSSE": _calc_rmsse,
    "MAPE": _calc_mape,
    "SMAPE": _calc_smape,
}

