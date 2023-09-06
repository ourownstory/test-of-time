import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer


class LogTransformer(FunctionTransformer):
    def __init__(self):
        self.epsilon = 1e-8
        super().__init__(self._log_transform, self._exp_transform, validate=False)

    def _log_transform(self, X):
        return np.log1p(np.maximum(X, self.epsilon))

    def _exp_transform(self, X):
        return np.expm1(X)


class ShiftedBoxCoxTransformer(PowerTransformer):
    def __init__(self):
        super().__init__(method="box-cox", standardize=True)
        self.shift = None

    def fit(self, X, y=None):
        self.shift = abs(np.min(X)) + 1e-8
        return super().fit(X + self.shift, y)

    def transform(self, X):
        assert self.shift is not None, "Transformer must be fitted before calling transform"
        return super().transform(X + np.full(X.shape, self.shift))

    def inverse_transform(self, X):
        assert self.shift is not None, "Transformer must be fitted before calling inverse_transform"
        return super().inverse_transform(X) - np.full(X.shape, self.shift)
