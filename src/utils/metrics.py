import numpy as np


def _validate_shapes(y_true, y_pred):
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError(
            f"Shape mismatch: y_true {np.shape(y_true)} vs y_pred {np.shape(y_pred)}"
        )


def mae(y_true, y_pred):
    _validate_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    _validate_shapes(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))