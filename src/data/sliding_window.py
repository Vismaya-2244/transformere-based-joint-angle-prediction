import numpy as np


def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    past_steps: int = 128,
    future_steps: int = 20,
    future_start_step: int = 1,
    stride: int = 1,
):
    """
    Create seq-to-seq sliding windows.

    Input window:
        X_window = X[t - past_steps + 1 : t + 1]

    Target window:
        y_window = y[t + future_start_step : t + future_start_step + future_steps]

    Args:
        X: shape (T, input_features)
        y: shape (T,) or (T, output_features)
        past_steps: number of past time-steps in encoder input
        future_steps: number of future time-steps to predict
        future_start_step: first prediction step ahead (e.g., 2 -> t+20ms at 100Hz)
        stride: step size between consecutive windows

    Returns:
        X_windows: (N, past_steps, input_features)
        y_windows: (N, future_steps, output_features)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, F). Got shape: {X.shape}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim != 2:
        raise ValueError(f"y must be 1D or 2D. Got shape: {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")

    if past_steps <= 0 or future_steps <= 0 or future_start_step <= 0:
        raise ValueError("past_steps, future_steps, and future_start_step must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    # Latest index inside each past window is t_end.
    # We need y up to t_end + future_start_step + future_steps - 1
    min_total_len = past_steps + future_start_step + future_steps - 1
    if len(X) < min_total_len:
        raise ValueError(
            "Time series too short for given window settings: "
            f"need at least {min_total_len}, got {len(X)}"
        )

    # Number of valid window starts (with stride)
    max_start = len(X) - (past_steps + future_start_step + future_steps - 1)
    starts = np.arange(0, max_start, stride, dtype=np.int64)

    num_samples = len(starts)
    in_feats = X.shape[1]
    out_feats = y.shape[1]

    X_windows = np.zeros((num_samples, past_steps, in_feats), dtype=np.float32)
    y_windows = np.zeros((num_samples, future_steps, out_feats), dtype=np.float32)

    for j, i in enumerate(starts):
        # Past: i ... i+past_steps-1
        X_windows[j] = X[i : i + past_steps]

        # Future starts after past window by future_start_step
        y_start = i + past_steps - 1 + future_start_step
        y_end = y_start + future_steps
        y_windows[j] = y[y_start:y_end]

    return X_windows, y_windows