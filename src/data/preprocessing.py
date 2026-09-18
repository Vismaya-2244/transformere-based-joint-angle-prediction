import numpy as np


EPS = 1e-8


def _ensure_2d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D. Got shape {x.shape}")
    return x


def _ensure_1d_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1D or (T, 1). Got shape {y.shape}")
    return y


def _interpolate_nans_1d(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32, copy=True)
    n = len(signal)
    if n == 0:
        return signal

    mask = np.isnan(signal)
    if not mask.any():
        return signal

    valid_idx = np.where(~mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(signal, dtype=np.float32)

    signal[mask] = np.interp(np.where(mask)[0], valid_idx, signal[valid_idx])
    return signal


def fill_missing_multichannel(x: np.ndarray) -> np.ndarray:
    x = _ensure_2d(x, "x")
    filled = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        filled[:, c] = _interpolate_nans_1d(x[:, c])
    return filled


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size <= 1:
        return signal.astype(np.float32, copy=True)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def moving_average_multichannel(x: np.ndarray, window_size: int = 5) -> np.ndarray:
    x = _ensure_2d(x, "x")
    if window_size <= 1:
        return x.astype(np.float32, copy=True)

    smoothed = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        smoothed[:, c] = moving_average(x[:, c], window_size=window_size)
    return smoothed


def clip_outliers(x: np.ndarray, z_thresh: float = 4.0) -> np.ndarray:
    x = _ensure_2d(x, "x")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < EPS, 1.0, std)

    lower = mean - z_thresh * std
    upper = mean + z_thresh * std
    return np.clip(x, lower, upper).astype(np.float32)


def _zscore(x: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """
    Z-score normalize per channel.
    If mean/std are provided, reuse them.
    """
    x = _ensure_2d(x, "x")

    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)

    std = np.where(std < EPS, 1.0, std)
    x_norm = (x - mean) / (std + EPS)
    return x_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def synchronize_signals(imu: np.ndarray, emg: np.ndarray, labels: np.ndarray):
    """
    Synchronize by trimming all signals to the shortest length.
    Expects:
      imu:    (T1, imu_feats)
      emg:    (T2, emg_feats)
      labels: (T3,) or (T3, 1)
    """
    imu = _ensure_2d(imu, "imu")
    emg = _ensure_2d(emg, "emg")
    labels = _ensure_1d_labels(labels)

    min_len = min(len(imu), len(emg), len(labels))
    if min_len <= 0:
        raise ValueError("Synchronized length is zero. Check input arrays.")

    imu_sync = imu[:min_len]
    emg_sync = emg[:min_len]
    labels_sync = labels[:min_len]
    return imu_sync, emg_sync, labels_sync


def preprocess_imu(
    imu: np.ndarray,
    smooth_window: int = 3,
    clip_z: float = 4.0,
) -> np.ndarray:
    """
    Basic IMU preprocessing:
      1) fill missing values
      2) clip outliers
      3) light smoothing
    """
    imu = fill_missing_multichannel(imu)
    imu = clip_outliers(imu, z_thresh=clip_z)
    imu = moving_average_multichannel(imu, window_size=smooth_window)
    return imu.astype(np.float32)


def preprocess_emg(
    emg: np.ndarray,
    smooth_window: int = 5,
    rectify: bool = True,
    clip_z: float = 4.0,
) -> np.ndarray:
    """
    Basic EMG preprocessing:
      1) fill missing values
      2) optional full-wave rectification
      3) clip outliers
      4) smoothing
    """
    emg = fill_missing_multichannel(emg)
    emg = np.abs(emg) if rectify else emg.copy()
    emg = clip_outliers(emg, z_thresh=clip_z)
    emg = moving_average_multichannel(emg, window_size=smooth_window)
    return emg.astype(np.float32)


def synchronize_and_normalize(
    imu: np.ndarray,
    emg: np.ndarray,
    labels: np.ndarray,
    *,
    imu_stats=None,
    emg_stats=None,
    emg_rectify: bool = True,
    emg_smooth_window: int = 5,
    imu_smooth_window: int = 3,
    label_smooth_window: int = 1,
    imu_clip_z: float = 4.0,
    emg_clip_z: float = 4.0,
):
    """
    Full preprocessing pipeline:
    1) synchronize to common timeline
    2) preprocess IMU and EMG
    3) optional label smoothing
    4) normalize IMU and EMG per channel
    5) concatenate into X = [IMU | EMG]

    Returns:
      X     : (T, imu_feats + emg_feats)
      y     : (T,)
      stats : dict with normalization params for reuse
    """
    imu_sync, emg_sync, y = synchronize_signals(imu, emg, labels)

    imu_proc = preprocess_imu(
        imu_sync,
        smooth_window=imu_smooth_window,
        clip_z=imu_clip_z,
    )

    emg_proc = preprocess_emg(
        emg_sync,
        smooth_window=emg_smooth_window,
        rectify=emg_rectify,
        clip_z=emg_clip_z,
    )

    y = _interpolate_nans_1d(y)
    if label_smooth_window > 1:
        y = moving_average(y, window_size=label_smooth_window)

    imu_mean, imu_std = (imu_stats if imu_stats is not None else (None, None))
    emg_mean, emg_std = (emg_stats if emg_stats is not None else (None, None))

    imu_norm, imu_mean, imu_std = _zscore(imu_proc, imu_mean, imu_std)
    emg_norm, emg_mean, emg_std = _zscore(emg_proc, emg_mean, emg_std)

    X = np.hstack([imu_norm, emg_norm]).astype(np.float32)
    y = y.astype(np.float32)

    stats = {
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "emg_mean": emg_mean,
        "emg_std": emg_std,
    }
    return X, y, stats
